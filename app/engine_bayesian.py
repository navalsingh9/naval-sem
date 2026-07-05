"""
NAVAL-SEM Bayesian Statistical Engine (v1.1, A8/A9)
====================================================
Bayesian estimation for CB-SEM measurement + structural specifications,
via a hand-rolled Metropolis-within-Gibbs sampler with data augmentation
for latent factor scores.

Why hand-rolled instead of PyMC
--------------------------------
This app ships as a PyInstaller desktop bundle (see naval_sem.spec) with a
fully pinned dependency set (pyproject.toml / uv.lock) that does not include
any probabilistic-programming library. PyMC pulls in PyTensor, which needs
runtime code generation and (on some platforms) a C/C++ toolchain — a real
packaging risk for a frozen build that doesn't already carry that machinery.
Given the "conservative MVP" mandate for this feature, we instead build the
sampler from numpy/scipy, both already hard dependencies.

Model class supported (v1.1 MVP scope — read before extending)
----------------------------------------------------------------
  * Single-group, continuous indicators, Gaussian measurement/structural
    errors (the standard CB-SEM assumption; matches what engine.py's ML
    path already assumes).
  * Reflective measurement only (parsed_model["measurement"], =~). Formative
    (<~ / Mode B) constructs are NOT supported here — raises ValueError.
    PLS-style formative Bayesian estimation is a materially different model
    (no reflective error structure to place a likelihood on) and is out of
    scope for this MVP.
  * No cross-loadings: each indicator belongs to exactly one latent
    variable, matching parse_lavaan()'s measurement dict structure.
  * No mean structure: all modeled columns are demeaned internally, so
    intercepts are implicitly 0. Only (co)variance-structure parameters
    (loadings, structural paths, variances) are estimated, matching what
    the priors spec (A9) and BayesianResult (A10) parameter set expect.
  * Marker-variable identification: for every latent variable, the first
    indicator listed in its measurement block has its loading fixed to
    1.0 and is not sampled. This mirrors semopy's own identification
    convention (see engine.py's `_extract_loadings` comment), so Bayesian
    and ML point estimates for the same model are directly comparable.
  * Single-indicator latent variables are perfectly-measured by
    convention (loading = 1, residual variance fixed ~0) since a single
    indicator cannot separately identify measurement error from factor
    variance. A log_fn "warn" is emitted when this happens.
  * ``~~`` covariance parameters (parsed_model["covariances"]) are NOT
    supported in this MVP. Modeling a non-zero covariance between two
    factors/disturbances correctly requires sampling their joint
    distribution (an Inverse-Wishart block) rather than the independent
    per-construct conjugate updates used here; doing that only for the
    reported number while sampling under an independence assumption would
    be statistically inconsistent, so we raise a clear, actionable
    ValueError instead of silently producing a misleading estimate.
  * Missing data: requires a complete-case DataFrame. Do listwise deletion
    (or another imputation) upstream, same convention as elsewhere in this
    codebase — this function does not implement FIML.

Parameter naming
-----------------
Every free parameter is keyed by a canonical lavaan-style string built the
same way build_semopy_syntax() and PathParameter already do:
  loadings          "{lv}=~{indicator}"
  structural paths  "{lhs}~{rhs}"
  factor / disturbance variance   "{name}~~{name}"
  indicator residual variance     "{indicator}~~{indicator}"
This is the key format the ``priors`` dict (A9) and BayesianResult.parameters
(A10) both use.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.engine_utils import _emit, _safe_float
from app.schemas import BayesianResult, BayesianParameterEntry, ParameterPosteriorDensity

logger = logging.getLogger("naval_sem.engine_bayesian")

_EPS_SINGLE_INDICATOR_VAR = 1e-6   # fixed residual variance for single-indicator LVs
_DEFAULT_LOADING_PRIOR = {"type": "normal", "mean": 0.0, "sd": 10.0}   # A9 default
_DEFAULT_PATH_PRIOR = {"type": "normal", "mean": 0.0, "sd": 10.0}      # A9 default
# Variance parameters cannot use A9's literal Normal(0, 10) default — a
# Normal prior places positive density on negative variances, which is not
# a valid model. We use a weakly-informative Inverse-Gamma(2, 2) instead
# (mean undefined at shape=2's boundary... actually mean = b/(a-1) = 2, a
# common weakly-informative default for scale parameters in this size
# range, e.g. Gelman 2006). This deviation from the literal Normal(0,10)
# default is deliberate and documented here for methods-section accuracy;
# see fit_bayesian_sem's docstring for the full explanation.
_DEFAULT_VAR_PRIOR = {"type": "inverse_gamma", "shape": 2.0, "scale": 2.0}

_RHAT_CONVERGENCE_THRESHOLD = 1.01


# ═══════════════════════════════════════════════════════════════════════════
# ── Parameter registry ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _build_parameter_registry(parsed_model: dict, log_fn: Optional[Callable] = None) -> dict:
    """
    Translate parse_lavaan() output into the bookkeeping structures the
    sampler needs: which loadings/paths/variances are free parameters, which
    latent variables get their factor scores sampled vs. fixed to a raw
    column, and the canonical name string for every free parameter.

    Returns a dict with keys:
      measurement, struct_by_lhs, latent_vars, exogenous_lvs,
      marker_of (lv -> marker indicator name),
      free_loadings (list of (lv, indicator)),
      fixed_single_indicator_lvs (set),
      free_indicator_vars (list of indicator names needing a residual-var param),
      free_construct_vars (list of construct names needing a factor/disturbance-var param),
      predictors_of (lhs -> list[rhs]), dependents_of (rhs -> list[(lhs, rhs)])
    """
    measurement: dict = parsed_model.get("measurement", {})
    structural: list = parsed_model.get("structural", [])
    covariances: list = parsed_model.get("covariances", [])
    latent_vars: list = list(parsed_model.get("latent_vars", []))
    construct_modes: dict = parsed_model.get("construct_modes", {})

    formative = [lv for lv, mode in construct_modes.items() if mode == "B"]
    if formative:
        raise ValueError(
            "fit_bayesian_sem: formative (Mode B, '<~') constructs are not "
            f"supported in the v1.1 Bayesian MVP: {formative}. Bayesian "
            "estimation here assumes a reflective measurement likelihood; "
            "formative constructs have no such likelihood to place a prior "
            "against. Switch these to reflective (=~) or use the CB-SEM/"
            "PLS-SEM engine instead."
        )

    if covariances:
        cov_names = [f"{c['lhs']}~~{c['rhs']}" for c in covariances]
        raise ValueError(
            "fit_bayesian_sem: explicit covariance ('~~') terms are not "
            f"supported in the v1.1 Bayesian MVP: {cov_names}. Correctly "
            "modeling a non-zero covariance between two factors or "
            "disturbances requires jointly sampling them (an Inverse-"
            "Wishart block) rather than the independent per-construct "
            "conjugate updates this sampler uses; reporting a covariance "
            "number while sampling under independence would be "
            "statistically inconsistent. Remove the covariance line(s) to "
            "run this model, or use the ML/PLS engine, which supports them."
        )

    marker_of: dict[str, str] = {}
    free_loadings: list[tuple[str, str]] = []
    fixed_single_indicator_lvs: set = set()
    free_indicator_vars: list[str] = []

    for lv, indicators in measurement.items():
        if not indicators:
            continue
        marker_of[lv] = indicators[0]
        if len(indicators) == 1:
            fixed_single_indicator_lvs.add(lv)
            _emit(log_fn, "warn",
                  f"Latent variable '{lv}' has a single indicator "
                  f"('{indicators[0]}'). Treating it as perfectly measured "
                  "(loading=1, residual variance fixed near 0) — a single "
                  "indicator cannot separately identify measurement error "
                  "from factor variance.")
            continue
        for ind in indicators[1:]:
            free_loadings.append((lv, ind))
        for ind in indicators:
            free_indicator_vars.append(ind)

    struct_by_lhs: dict[str, list[str]] = {}
    for rel in structural:
        struct_by_lhs.setdefault(rel["lhs"], []).append(rel["rhs"])

    exogenous_lvs = [lv for lv in latent_vars if lv not in struct_by_lhs]

    # free_construct_vars: every construct that needs its own variance param
    # — exogenous latents (factor variance) + every regression outcome
    # (disturbance variance), latent or observed.
    free_construct_vars = list(exogenous_lvs) + list(struct_by_lhs.keys())

    dependents_of: dict[str, list[tuple[str, str]]] = {}
    for lhs, preds in struct_by_lhs.items():
        for rhs in preds:
            dependents_of.setdefault(rhs, []).append((lhs, rhs))

    # sampled_etas: latent vars whose factor score is actually Gibbs-sampled
    # (has a measurement block with >=2 indicators). Single-indicator LVs and
    # plain observed variables are NOT sampled — their "construct value" is
    # just the (demeaned) data column.
    sampled_etas = [lv for lv in latent_vars if lv not in fixed_single_indicator_lvs]

    return dict(
        measurement=measurement,
        struct_by_lhs=struct_by_lhs,
        latent_vars=latent_vars,
        exogenous_lvs=exogenous_lvs,
        marker_of=marker_of,
        free_loadings=free_loadings,
        fixed_single_indicator_lvs=fixed_single_indicator_lvs,
        free_indicator_vars=free_indicator_vars,
        free_construct_vars=free_construct_vars,
        dependents_of=dependents_of,
        sampled_etas=sampled_etas,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── Priors ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_prior(name: str, priors: Optional[dict], is_variance: bool) -> dict:
    """Return the resolved prior spec dict for parameter `name`."""
    if priors and name in priors:
        spec = dict(priors[name])
        t = spec.get("type")
        if t not in ("normal", "uniform", "custom"):
            raise ValueError(
                f"fit_bayesian_sem: prior for '{name}' has unsupported "
                f"type '{t}'. Use 'normal', 'uniform', or 'custom'."
            )
        if t == "custom":
            samples = np.asarray(spec.get("samples", []), dtype=float)
            if samples.size < 10:
                raise ValueError(
                    f"fit_bayesian_sem: custom prior for '{name}' needs "
                    "at least 10 'samples' to fit an empirical density."
                )
            spec["_kde"] = stats.gaussian_kde(samples)
            spec["_lo"] = float(samples.min())
            spec["_hi"] = float(samples.max())
        if is_variance and t == "uniform" and spec.get("low", 0) <= 0:
            spec["low"] = max(spec.get("low", 0.0), 1e-8)
        return spec
    return dict(_DEFAULT_VAR_PRIOR if is_variance else _DEFAULT_LOADING_PRIOR)


def _prior_logpdf(spec: dict, x: float) -> float:
    """Log-density of value x under a resolved (non-conjugate) prior spec."""
    t = spec["type"]
    if t == "normal":
        return float(stats.norm.logpdf(x, loc=spec["mean"], scale=spec["sd"]))
    if t == "uniform":
        lo, hi = spec["low"], spec["high"]
        return 0.0 if lo <= x <= hi else -np.inf
    if t == "custom":
        if x < spec["_lo"] or x > spec["_hi"]:
            return -np.inf
        d = float(spec["_kde"].evaluate([x])[0])
        return np.log(d) if d > 0 else -np.inf
    if t == "inverse_gamma":
        if x <= 0:
            return -np.inf
        return float(stats.invgamma.logpdf(x, a=spec["shape"], scale=spec["scale"]))
    raise ValueError(f"Unknown prior type '{t}'")


def _is_conjugate_normal(spec: dict) -> bool:
    return spec["type"] == "normal"


def _is_default_inverse_gamma(spec: dict) -> bool:
    return spec["type"] == "inverse_gamma"


# ═══════════════════════════════════════════════════════════════════════════
# ── Adaptive random-walk Metropolis (for non-conjugate priors) ─────────────
# ═══════════════════════════════════════════════════════════════════════════

class _AdaptiveMH:
    """Single-parameter random-walk Metropolis with warmup-only step-size
    adaptation (target acceptance ~0.35), frozen once warmup ends — a
    standard, simple, detailed-balance-preserving adaptation scheme."""

    def __init__(self, init_value: float, transform: str = "identity"):
        self.value = init_value
        self.step = 0.5 if transform == "identity" else 0.3
        self.transform = transform   # "identity" or "log" (for positivity)
        self._accepts = 0
        self._trials = 0

    def step_update(self, rng: np.random.Generator, log_target_fn, in_warmup: bool) -> float:
        cur = self.value
        cur_t = np.log(cur) if self.transform == "log" else cur
        prop_t = cur_t + rng.normal(0.0, self.step)
        prop = np.exp(prop_t) if self.transform == "log" else prop_t

        log_cur = log_target_fn(cur)
        log_prop = log_target_fn(prop) if prop > 0 or self.transform == "identity" else -np.inf
        # Jacobian for log-transform proposals cancels in a symmetric RW on
        # the log scale only if we also account for it; a RW on log(x) with
        # symmetric increments corresponds to proposal density q(x'|x) =
        # Normal(log x, step) / x' — the 1/x' Jacobian term must be included.
        if self.transform == "log" and prop > 0:
            log_prop = log_prop + np.log(prop) - np.log(cur)   # Hastings correction

        if log_cur == -np.inf:
            # Current state is out of prior support (can happen transiently
            # at a variance boundary under float precision) — accept any
            # in-support proposal to escape rather than propagate a NaN.
            accept = log_prop > -np.inf
        elif log_prop == -np.inf:
            accept = False
        else:
            log_alpha = log_prop - log_cur
            accept = np.log(rng.uniform()) < log_alpha
        self._trials += 1
        if accept:
            self.value = prop
            self._accepts += 1
        if in_warmup and self._trials % 50 == 0:
            rate = self._accepts / self._trials
            if rate < 0.25:
                self.step *= 0.8
            elif rate > 0.45:
                self.step *= 1.25
        return self.value


# ═══════════════════════════════════════════════════════════════════════════
# ── One MCMC chain ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _run_one_chain(
    data: dict[str, np.ndarray],
    reg: dict,
    priors: Optional[dict],
    n_iter: int,
    n_warmup: int,
    seed: int,
    n: int,
    log_fn: Optional[Callable] = None,
    chain_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Run a single chain of the Metropolis-within-Gibbs sampler. Returns
    {param_name: np.ndarray of shape (n_iter - n_warmup,)} post-warmup draws."""
    rng = np.random.default_rng(seed)

    measurement = reg["measurement"]
    struct_by_lhs = reg["struct_by_lhs"]
    marker_of = reg["marker_of"]
    dependents_of = reg["dependents_of"]
    sampled_etas = reg["sampled_etas"]
    fixed_single = reg["fixed_single_indicator_lvs"]

    def get_val(name: str) -> np.ndarray:
        """Current value vector (n,) for any construct: sampled latent
        factor score, or fixed data column (single-indicator LV or plain
        observed variable)."""
        if name in eta:
            return eta[name]
        return data[name]

    # ── Initialize state ────────────────────────────────────────────────────
    eta: dict[str, np.ndarray] = {}
    for lv in sampled_etas:
        # init from mean of its indicators (a sane, cheap starting point)
        cols = [data[i] for i in measurement[lv]]
        eta[lv] = np.mean(cols, axis=0) + rng.normal(0, 0.1, size=n)
    for lv in fixed_single:
        eta[lv] = data[marker_of[lv]]   # fixed identity, never resampled

    lam: dict[str, float] = {ind: 1.0 for lv, ind in reg["free_loadings"]}
    psi: dict[str, float] = {ind: 1.0 for ind in reg["free_indicator_vars"]}
    beta: dict[tuple[str, str], float] = {
        (lhs, rhs): 0.0 for lhs, preds in struct_by_lhs.items() for rhs in preds
    }
    phi: dict[str, float] = {c: 1.0 for c in reg["free_construct_vars"]}

    # ── Resolve priors + set up MH samplers for non-conjugate parameters ──
    loading_priors = {(lv, ind): _resolve_prior(f"{lv}=~{ind}", priors, is_variance=False)
                       for lv, ind in reg["free_loadings"]}
    path_priors = {(lhs, rhs): _resolve_prior(f"{lhs}~{rhs}", priors, is_variance=False)
                   for lhs, preds in struct_by_lhs.items() for rhs in preds}
    indvar_priors = {ind: _resolve_prior(f"{ind}~~{ind}", priors, is_variance=True)
                      for ind in reg["free_indicator_vars"]}
    constructvar_priors = {c: _resolve_prior(f"{c}~~{c}", priors, is_variance=True)
                            for c in reg["free_construct_vars"]}

    mh_loadings = {k: _AdaptiveMH(1.0) for k, sp in loading_priors.items()
                   if not _is_conjugate_normal(sp)}
    # A structural equation is sampled via per-coefficient MH (rather than
    # the joint conjugate block update) if ANY of its predictors has a
    # non-normal prior — so every coefficient in that equation needs an MH
    # sampler, including ones whose own prior happens to be normal.
    _mixed_equations = {
        lhs for lhs, preds in struct_by_lhs.items()
        if not all(_is_conjugate_normal(path_priors[(lhs, rhs)]) for rhs in preds)
    }
    mh_paths = {(lhs, rhs): _AdaptiveMH(0.0)
                for lhs, preds in struct_by_lhs.items() if lhs in _mixed_equations
                for rhs in preds}
    mh_indvars = {k: _AdaptiveMH(1.0, transform="log") for k, sp in indvar_priors.items()
                  if not _is_default_inverse_gamma(sp)}
    mh_constructvars = {k: _AdaptiveMH(1.0, transform="log") for k, sp in constructvar_priors.items()
                        if not _is_default_inverse_gamma(sp)}

    n_keep = n_iter - n_warmup
    draws: dict[str, np.ndarray] = {}
    for lv, ind in reg["free_loadings"]:
        draws[f"{lv}=~{ind}"] = np.empty(n_keep)
    for lhs, preds in struct_by_lhs.items():
        for rhs in preds:
            draws[f"{lhs}~{rhs}"] = np.empty(n_keep)
    for ind in reg["free_indicator_vars"]:
        draws[f"{ind}~~{ind}"] = np.empty(n_keep)
    for c in reg["free_construct_vars"]:
        draws[f"{c}~~{c}"] = np.empty(n_keep)

    _t0 = time.time()
    _report_every = max(1, n_iter // 10)

    for it in range(n_iter):
        in_warmup = it < n_warmup

        # ── Step 1: latent factor scores η (systematic-scan Gibbs) ────────
        for k in sampled_etas:
            precision = 0.0
            mean_num = np.zeros(n)
            for ind in measurement[k]:
                lam_j = 1.0 if ind == marker_of[k] else lam[ind]
                psi_j = psi.get(ind, _EPS_SINGLE_INDICATOR_VAR)
                precision += (lam_j ** 2) / psi_j
                mean_num += (lam_j / psi_j) * data[ind]

            if k in struct_by_lhs:
                phi_k = phi[k]
                precision += 1.0 / phi_k
                pred_sum = np.zeros(n)
                for rhs in struct_by_lhs[k]:
                    pred_sum += beta[(k, rhs)] * get_val(rhs)
                mean_num += (1.0 / phi_k) * pred_sum
            else:
                phi_k = phi[k]
                precision += 1.0 / phi_k
                # mean contribution is 0 (exogenous, prior mean 0)

            for (m, rhs_k) in dependents_of.get(k, []):
                beta_mk = beta[(m, rhs_k)]
                phi_m = phi[m]
                other_pred_sum = np.zeros(n)
                for rhs2 in struct_by_lhs[m]:
                    if rhs2 == k:
                        continue
                    other_pred_sum += beta[(m, rhs2)] * get_val(rhs2)
                resid_m = get_val(m) - other_pred_sum
                precision += (beta_mk ** 2) / phi_m
                mean_num += (beta_mk / phi_m) * resid_m

            var_k = 1.0 / precision
            mean_k = var_k * mean_num
            eta[k] = rng.normal(mean_k, np.sqrt(var_k))

        # ── Step 2: loadings λ ──────────────────────────────────────────────
        for (lv, ind) in reg["free_loadings"]:
            eta_lv = eta[lv]
            psi_j = psi[ind]
            spec = loading_priors[(lv, ind)]
            if _is_conjugate_normal(spec):
                prior_prec = 1.0 / spec["sd"] ** 2
                prec_post = prior_prec + float(eta_lv @ eta_lv) / psi_j
                mean_post = (spec["mean"] * prior_prec +
                             float(eta_lv @ data[ind]) / psi_j) / prec_post
                lam[ind] = rng.normal(mean_post, np.sqrt(1.0 / prec_post))
            else:
                resid_const = data[ind]

                def log_target(x, _eta=eta_lv, _y=resid_const, _psi=psi_j, _spec=spec):
                    resid = _y - x * _eta
                    ll = -0.5 * np.sum(resid ** 2) / _psi
                    return ll + _prior_logpdf(_spec, x)
                lam[ind] = mh_loadings[(lv, ind)].step_update(rng, log_target, in_warmup)

        # ── Step 3: indicator residual variances ψ ──────────────────────────
        for ind in reg["free_indicator_vars"]:
            lv = _find_lv_of_indicator(measurement, ind)
            lam_j = 1.0 if ind == marker_of[lv] else lam[ind]
            resid = data[ind] - lam_j * eta[lv]
            spec = indvar_priors[ind]
            if _is_default_inverse_gamma(spec):
                a_post = spec["shape"] + n / 2.0
                b_post = spec["scale"] + 0.5 * float(resid @ resid)
                psi[ind] = 1.0 / rng.gamma(a_post, 1.0 / b_post)
            else:
                def log_target(x, _resid=resid, _spec=spec):
                    if x <= 0:
                        return -np.inf
                    ll = -0.5 * n * np.log(x) - 0.5 * float(_resid @ _resid) / x
                    return ll + _prior_logpdf(_spec, x)
                psi[ind] = mh_indvars[ind].step_update(rng, log_target, in_warmup)

        # ── Step 4: structural path coefficients β (per-equation block) ────
        for lhs, preds in struct_by_lhs.items():
            y = get_val(lhs)
            phi_lhs = phi[lhs]
            specs = [path_priors[(lhs, rhs)] for rhs in preds]
            if all(_is_conjugate_normal(s) for s in specs):
                X = np.column_stack([get_val(rhs) for rhs in preds])
                m0 = np.array([s["mean"] for s in specs])
                sd0 = np.array([s["sd"] for s in specs])
                prior_prec = np.diag(1.0 / sd0 ** 2)
                prec_post = prior_prec + (X.T @ X) / phi_lhs
                b_vec = (m0 / sd0 ** 2) + (X.T @ y) / phi_lhs
                cov_post = np.linalg.inv(prec_post)
                mean_post = cov_post @ b_vec
                sample = rng.multivariate_normal(mean_post, cov_post)
                for rhs, val in zip(preds, sample):
                    beta[(lhs, rhs)] = float(val)
            else:
                # Documented MVP simplification: if any predictor in this
                # equation has a non-normal prior, fall back to per-
                # coefficient random-walk Metropolis for the whole equation
                # rather than a mixed joint-conjugate/MH block sampler.
                for rhs in preds:
                    other_sum = np.zeros(n)
                    for rhs2 in preds:
                        if rhs2 == rhs:
                            continue
                        other_sum += beta[(lhs, rhs2)] * get_val(rhs2)
                    resid_const = y - other_sum
                    x_rhs = get_val(rhs)
                    spec = path_priors[(lhs, rhs)]

                    def log_target(b, _resid=resid_const, _x=x_rhs, _phi=phi_lhs, _spec=spec):
                        resid = _resid - b * _x
                        ll = -0.5 * np.sum(resid ** 2) / _phi
                        return ll + _prior_logpdf(_spec, b)
                    beta[(lhs, rhs)] = mh_paths[(lhs, rhs)].step_update(rng, log_target, in_warmup)

        # ── Step 5: construct (factor / disturbance) variances φ ───────────
        for c in reg["free_construct_vars"]:
            if c in struct_by_lhs:
                y = get_val(c)
                pred_sum = np.zeros(n)
                for rhs in struct_by_lhs[c]:
                    pred_sum += beta[(c, rhs)] * get_val(rhs)
                resid = y - pred_sum
            else:
                resid = get_val(c)   # exogenous: deviation from prior mean 0
            spec = constructvar_priors[c]
            if _is_default_inverse_gamma(spec):
                a_post = spec["shape"] + n / 2.0
                b_post = spec["scale"] + 0.5 * float(resid @ resid)
                phi[c] = 1.0 / rng.gamma(a_post, 1.0 / b_post)
            else:
                def log_target(x, _resid=resid, _spec=spec):
                    if x <= 0:
                        return -np.inf
                    ll = -0.5 * n * np.log(x) - 0.5 * float(_resid @ _resid) / x
                    return ll + _prior_logpdf(_spec, x)
                phi[c] = mh_constructvars[c].step_update(rng, log_target, in_warmup)

        # ── Record post-warmup draws ────────────────────────────────────────
        if not in_warmup:
            i_keep = it - n_warmup
            for lv, ind in reg["free_loadings"]:
                draws[f"{lv}=~{ind}"][i_keep] = lam[ind]
            for lhs, preds in struct_by_lhs.items():
                for rhs in preds:
                    draws[f"{lhs}~{rhs}"][i_keep] = beta[(lhs, rhs)]
            for ind in reg["free_indicator_vars"]:
                draws[f"{ind}~~{ind}"][i_keep] = psi[ind]
            for c in reg["free_construct_vars"]:
                draws[f"{c}~~{c}"][i_keep] = phi[c]

        if log_fn is not None and (it + 1) % _report_every == 0:
            elapsed = time.time() - _t0
            _emit(log_fn, "info",
                  f"  Chain {chain_idx + 1}: {it + 1}/{n_iter} iterations "
                  f"({'warmup' if in_warmup else 'sampling'}) · {elapsed:.1f}s elapsed")

    return draws


def _find_lv_of_indicator(measurement: dict, ind: str) -> str:
    for lv, inds in measurement.items():
        if ind in inds:
            return lv
    raise ValueError(f"Indicator '{ind}' not found in measurement block.")


# ═══════════════════════════════════════════════════════════════════════════
# ── Diagnostics: split R-hat, rank-normalized bulk-ESS, HPD interval ───────
# ═══════════════════════════════════════════════════════════════════════════

def _split_chains(chain_draws: np.ndarray) -> np.ndarray:
    """chain_draws: (n_chains, n_samples) -> (2*n_chains, n_samples//2), each
    chain split in half (Gelman et al. 2013's recommended split-R-hat)."""
    n_chains, n_samples = chain_draws.shape
    half = n_samples // 2
    first = chain_draws[:, :half]
    second = chain_draws[:, n_samples - half:]
    return np.concatenate([first, second], axis=0)


def _split_rhat(chain_draws: np.ndarray) -> float:
    """Gelman-Rubin split R-hat. chain_draws: (n_chains, n_samples)."""
    split = _split_chains(chain_draws)
    m, n = split.shape
    if m < 2 or n < 2:
        return float("nan")
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    if W <= 0:
        return float("inf") if var_hat > 0 else 1.0
    return float(np.sqrt(var_hat / W))


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normalize a flat sample array to approximate normal scores
    (Vehtari et al. 2021), used for a robust bulk-ESS estimate."""
    ranks = stats.rankdata(x, method="average")
    n = len(x)
    p = (ranks - 0.5) / n
    return stats.norm.ppf(p)


def _bulk_ess(chain_draws: np.ndarray) -> float:
    """Rank-normalized bulk effective sample size (Vehtari et al. 2021),
    via Geyer's initial monotone sequence estimator on the autocorrelation
    of rank-normalized, split chains."""
    split = _split_chains(chain_draws)
    m, n = split.shape
    if n < 4:
        return float(m * n)
    flat = split.reshape(-1)
    z = _rank_normalize(flat).reshape(m, n)

    # Average autocorrelation across chains via FFT (fast, standard).
    z_c = z - z.mean(axis=1, keepdims=True)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.fft(z_c, n=nfft, axis=1)
    acov = np.fft.ifft(f * np.conjugate(f), axis=1).real[:, :n]
    acov /= n
    var_within = acov[:, 0].mean()
    if var_within <= 0:
        return float(m * n)
    rho = acov.mean(axis=0) / var_within   # average over chains, lag 0..n-1
    rho[0] = 1.0

    # Geyer's initial positive sequence: sum paired lags while positive.
    tau = 1.0
    t = 1
    while t + 1 < n:
        pair_sum = rho[t] + rho[t + 1]
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
        t += 2
    ess = (m * n) / max(tau, 1e-8)
    return float(min(ess, m * n))


def _hpd_interval(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Shortest interval containing `mass` fraction of samples (Chen & Shao
    1999) — the standard HPD estimator, NOT a percentile interval."""
    s = np.sort(samples)
    n = len(s)
    window = int(np.ceil(mass * n))
    if window >= n:
        return float(s[0]), float(s[-1])
    widths = s[window:] - s[:n - window]
    start = int(np.argmin(widths))
    return float(s[start]), float(s[start + window])


def _posterior_density(samples: np.ndarray, n_bins: int = 50) -> tuple[list[float], list[float]]:
    """Binned posterior density (histogram) for one parameter: 50 bin
    centers (x) and density heights (y), per A11 — never raw samples."""
    counts, edges = np.histogram(samples, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return [float(c) for c in centers], [float(v) for v in counts]


# ═══════════════════════════════════════════════════════════════════════════
# ── Public entry points ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _fit_bayesian_sem_impl(
    df: pd.DataFrame,
    parsed_model: dict,
    priors: Optional[dict] = None,
    n_chains: int = 4,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    rng_seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> tuple[BayesianResult, dict[str, np.ndarray]]:
    _emit(log_fn, "step",
          f"Bayesian SEM: building parameter registry "
          f"({n_chains} chains × {n_samples} samples, {n_warmup} warmup)")
    reg = _build_parameter_registry(parsed_model, log_fn=log_fn)

    needed_cols = set(reg["free_indicator_vars"])
    for lv in reg["fixed_single_indicator_lvs"]:
        needed_cols.add(reg["marker_of"][lv])
    for lhs, preds in reg["struct_by_lhs"].items():
        needed_cols.add(lhs)
        needed_cols.update(preds)
    needed_cols = [c for c in needed_cols if c in df.columns]

    work = df[needed_cols].apply(pd.to_numeric, errors="coerce")
    if work.isna().any().any():
        n_before = len(work)
        work = work.dropna()
        _emit(log_fn, "warn",
              f"Bayesian SEM requires complete cases (no FIML support yet) — "
              f"dropped {n_before - len(work)} row(s) with missing values.")
    if len(work) < 10:
        raise ValueError("fit_bayesian_sem: fewer than 10 complete rows available to fit.")

    n = len(work)
    data = {c: (work[c].values.astype(float) - work[c].values.astype(float).mean())
            for c in needed_cols}

    _emit(log_fn, "info",
          f"Model structure: {len(reg['latent_vars'])} latent vars · "
          f"{len(reg['free_indicator_vars']) + len(reg['fixed_single_indicator_lvs'])} indicators · "
          f"{sum(len(v) for v in reg['struct_by_lhs'].values())} structural paths · n={n}")

    all_chain_draws: list[dict[str, np.ndarray]] = []
    for c_idx in range(n_chains):
        _emit(log_fn, "step", f"Bayesian SEM: running chain {c_idx + 1}/{n_chains}")
        chain_draws = _run_one_chain(
            data=data, reg=reg, priors=priors,
            n_iter=n_warmup + n_samples, n_warmup=n_warmup,
            seed=rng_seed + c_idx, n=n, log_fn=log_fn, chain_idx=c_idx,
        )
        all_chain_draws.append(chain_draws)

    param_names = list(all_chain_draws[0].keys())
    _emit(log_fn, "step", "Bayesian SEM: computing convergence diagnostics (R-hat, ESS, HPD)")

    parameters: list[BayesianParameterEntry] = []
    convergence_warnings: list[str] = []
    flat_samples: dict[str, np.ndarray] = {}

    for name in param_names:
        stacked = np.stack([cd[name] for cd in all_chain_draws], axis=0)  # (n_chains, n_samples)
        flat = stacked.reshape(-1)
        flat_samples[name] = flat

        r_hat = _split_rhat(stacked)
        ess = _bulk_ess(stacked)
        hpd_lo, hpd_hi = _hpd_interval(flat)

        if name in [f"{lv}=~{ind}" for lv, ind in reg["free_loadings"]]:
            op, lhs, rhs = "=~", *name.split("=~")
        elif "~~" in name:
            op = "~~"
            lhs, rhs = name.split("~~")
        else:
            op = "~"
            lhs, rhs = name.split("~", 1)

        parameters.append(BayesianParameterEntry(
            name=name, op=op, lhs=lhs, rhs=rhs,
            posterior_mean=_safe_float(float(np.mean(flat))),
            posterior_median=_safe_float(float(np.median(flat))),
            posterior_sd=_safe_float(float(np.std(flat, ddof=1))),
            hpd_lower=_safe_float(hpd_lo),
            hpd_upper=_safe_float(hpd_hi),
            r_hat=_safe_float(r_hat, default=float("nan")),
            ess_bulk=_safe_float(ess, default=0.0),
        ))
        if not (r_hat <= _RHAT_CONVERGENCE_THRESHOLD):
            convergence_warnings.append(name)

    converged = len(convergence_warnings) == 0
    if converged:
        _emit(log_fn, "ok", "Bayesian SEM: all parameters converged (max R-hat <= 1.01)")
    else:
        _emit(log_fn, "warn",
              f"Bayesian SEM: {len(convergence_warnings)} parameter(s) failed the "
              f"R-hat check: {convergence_warnings}")

    result = BayesianResult(
        parameters=parameters,
        n_chains=n_chains,
        n_samples=n_samples,
        converged=converged,
        convergence_warnings=convergence_warnings,
    )
    return result, flat_samples


def fit_bayesian_sem(
    df: pd.DataFrame,
    parsed_model: dict,
    priors: Optional[dict] = None,
    n_chains: int = 4,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    rng_seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> BayesianResult:
    """
    Fit a Bayesian CB-SEM (measurement + structural) model via Metropolis-
    within-Gibbs MCMC. See module docstring for supported model scope.

    Parameters
    ----------
    df : pd.DataFrame
        Data. Must contain complete cases for every column referenced by
        the model (no FIML support yet — see module docstring).
    parsed_model : dict
        Output of app.parser.parse_lavaan() — the same object engine.py's
        CB-SEM path already consumes.
    priors : Optional[dict[str, dict]]
        Keyed by canonical parameter name (see module docstring), each
        value {"type": "normal", "mean":, "sd":} | {"type": "uniform",
        "low":, "high":} | {"type": "custom", "samples": [...]}. Unspecified
        loading/path parameters default to Normal(0, 10); unspecified
        variance parameters default to InverseGamma(2, 2) (see
        _DEFAULT_VAR_PRIOR docstring for why Normal(0,10) is not used there).
    log_fn : Optional[Callable[[str, str], None]]
        Same (level, msg) progress-log callback used elsewhere in this
        codebase (see engine_utils._emit / main.py's _make_log_fn).

    Returns
    -------
    BayesianResult
    """
    result, _samples = _fit_bayesian_sem_impl(
        df, parsed_model, priors=priors, n_chains=n_chains,
        n_samples=n_samples, n_warmup=n_warmup, rng_seed=rng_seed, log_fn=log_fn,
    )
    return result


def fit_bayesian_sem_with_density(
    df: pd.DataFrame,
    parsed_model: dict,
    priors: Optional[dict] = None,
    n_chains: int = 4,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    rng_seed: int = 42,
    log_fn: Optional[Callable] = None,
    n_bins: int = 50,
) -> tuple[BayesianResult, list[ParameterPosteriorDensity]]:
    """
    Same as fit_bayesian_sem(), plus binned posterior-density points per
    parameter (A11) for the /bayesian-sem endpoint's response. Kept separate
    from fit_bayesian_sem() so that function's signature/return type exactly
    match the A8 spec.
    """
    result, flat_samples = _fit_bayesian_sem_impl(
        df, parsed_model, priors=priors, n_chains=n_chains,
        n_samples=n_samples, n_warmup=n_warmup, rng_seed=rng_seed, log_fn=log_fn,
    )
    density = []
    for p in result.parameters:
        x, y = _posterior_density(flat_samples[p.name], n_bins=n_bins)
        density.append(ParameterPosteriorDensity(name=p.name, x=x, y=y))
    return result, density
