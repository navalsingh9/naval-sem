"""
engine_lca.py  —  NAVAL-SEM v1.1 (A12–A15)
============================================
General-purpose latent class / finite mixture analysis.

This module deliberately reuses the EM scaffolding already proven in
fimix.py rather than re-implementing it:

  - ``_ols_paths``     (weighted OLS)         — reused for "mixture_regression"
  - ``_relative_entropy`` (R_E)                — reused verbatim for R_E
  - ``recommend_k``    (entropy-gated CAIC)    — reused verbatim (the "M2 fix")

Unlike FIMIX-PLS, which segments the *structural* (inner) model of a fitted
PLS-SEM, ``run_lca`` operates directly on raw indicator columns of the input
DataFrame — i.e. it is a standalone segmentation/mixture engine, not tied to
a lavaan-style structural model.

References
----------
Vermunt, J. K., & Magidson, J. (2002). Latent class cluster analysis.
Wedel, M., & DeSarbo, W. S. (1995). A mixture likelihood approach for
    generalized linear models. J. Classification, 12(1), 21-55.
Sarstedt et al. (2011) — entropy-gated K selection, reused via
    ``app.fimix.recommend_k``.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _emit, _safe_float
from app.fimix import _ols_paths, _relative_entropy, recommend_k
from app.schemas import LCAResult, LCAFitRow, LCAClassParameters

logger = logging.getLogger("naval_sem.engine_lca")

LCAMode = Literal["segmentation", "mixture_regression", "mixture_factor"]


# ── private helpers ──────────────────────────────────────────────────────────

def _validate_known_class_col(df: pd.DataFrame, known_class_col: str, K: int) -> np.ndarray:
    """
    Returns an (n,) int array with the fixed class index for labelled rows
    and -1 for unlabelled rows. Raises ValueError if labels aren't a subset
    of range(K).
    """
    if known_class_col not in df.columns:
        raise ValueError(f"LCA: known_class_col '{known_class_col}' not found in data.")
    raw = df[known_class_col]
    fixed = np.full(len(df), -1, dtype=int)
    non_null_mask = raw.notna()
    if non_null_mask.any():
        vals = raw[non_null_mask].astype(int)
        bad = sorted(int(v) for v in set(vals.unique()) - set(range(K)))
        if bad:
            raise ValueError(
                f"LCA: known_class_col values must be a subset of range(k)=0..{K - 1}; "
                f"found out-of-range value(s) {bad}."
            )
        fixed[non_null_mask.values] = vals.values
    return fixed


def _apply_fixed_labels(tau: np.ndarray, fixed_class: np.ndarray, K: int) -> np.ndarray:
    """Overwrite rows with a known label to a one-hot responsibility vector."""
    tau = tau.copy()
    labelled = fixed_class >= 0
    if labelled.any():
        tau[labelled, :] = 0.0
        tau[labelled, fixed_class[labelled]] = 1.0
    return tau


def _apply_equality_constraints(seg_params: list, class_weight_sums: np.ndarray,
                                 equality_constraints: Optional[list]) -> list:
    """
    Pool the weighted M-step update across classes for any constrained
    parameter name, then assign the pooled value back into every class's
    parameter set. Unconstrained parameters remain free per class.
    """
    if not equality_constraints:
        return seg_params
    total_w = float(class_weight_sums.sum())
    for name in equality_constraints:
        pooled, found_w = 0.0, 0.0
        for k, params in enumerate(seg_params):
            if name in params:
                pooled += class_weight_sums[k] * params[name]
                found_w += class_weight_sums[k]
        if found_w < 1e-12:
            continue
        pooled_val = pooled / max(total_w, 1e-12)
        for params in seg_params:
            if name in params:
                params[name] = float(pooled_val)
    return seg_params


def _mstep_segmentation(Xind: np.ndarray, w: np.ndarray, indicator_names: list) -> dict:
    wsum = w.sum()
    mu  = np.average(Xind, axis=0, weights=w)
    var = np.average((Xind - mu) ** 2, axis=0, weights=w)
    var = np.maximum(var, 1e-8)
    params = {}
    for name, m, v in zip(indicator_names, mu, var):
        params[f"mean_{name}"] = float(m)
        params[f"var_{name}"]  = float(v)
    return params


def _loglik_segmentation(Xind: np.ndarray, params: dict, indicator_names: list) -> np.ndarray:
    mu  = np.array([params[f"mean_{name}"] for name in indicator_names])
    var = np.maximum(np.array([params[f"var_{name}"] for name in indicator_names]), 1e-12)
    resid2 = (Xind - mu) ** 2
    lp = -0.5 * np.log(2 * np.pi * var) - 0.5 * resid2 / var
    return lp.sum(axis=1)


def _mstep_factor(Xind: np.ndarray, w: np.ndarray, indicator_names: list) -> dict:
    """Weighted single-factor model per class: mean + covariance -> top
    eigenpair gives loadings; residual diagonal gives uniquenesses."""
    wsum = max(w.sum(), 1e-12)
    mu = np.average(Xind, axis=0, weights=w)
    Xc = Xind - mu
    cov = (Xc.T * w) @ Xc / wsum
    cov = cov + np.eye(cov.shape[0]) * 1e-8   # numerical floor
    eigvals, eigvecs = np.linalg.eigh(cov)
    top = int(np.argmax(eigvals))
    lam1 = max(float(eigvals[top]), 0.0)
    v1 = eigvecs[:, top]
    loadings = v1 * np.sqrt(lam1)
    uniq = np.maximum(np.diag(cov) - loadings ** 2, 1e-6)
    params = {}
    for name, m, l, u in zip(indicator_names, mu, loadings, uniq):
        params[f"mean_{name}"]    = float(m)
        params[f"loading_{name}"] = float(l)
        params[f"uniq_{name}"]    = float(u)
    return params


def _loglik_factor(Xind: np.ndarray, params: dict, indicator_names: list) -> np.ndarray:
    mu   = np.array([params[f"mean_{name}"] for name in indicator_names])
    load = np.array([params[f"loading_{name}"] for name in indicator_names])
    uniq = np.maximum(np.array([params[f"uniq_{name}"] for name in indicator_names]), 1e-8)
    sigma = np.outer(load, load) + np.diag(uniq)
    sigma = sigma + np.eye(sigma.shape[0]) * 1e-8
    diff = Xind - mu
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        logdet = np.sum(np.log(uniq))   # fallback: treat as diagonal-only
        inv = np.diag(1.0 / uniq)
    else:
        inv = np.linalg.inv(sigma)
    p = Xind.shape[1]
    maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (p * np.log(2 * np.pi) + logdet + maha)


def _degenerate_params(mode: str, indicator_names: list, iv_names: Optional[list]) -> dict:
    """Fallback parameter set for a collapsed (near-zero-weight) class."""
    if mode == "segmentation":
        return {**{f"mean_{n}": 0.0 for n in indicator_names},
                **{f"var_{n}": 1.0 for n in indicator_names}}
    if mode == "mixture_regression":
        return {**{n: 0.0 for n in iv_names}, "sigma2": 1.0}
    if mode == "mixture_factor":
        return {**{f"mean_{n}": 0.0 for n in indicator_names},
                **{f"loading_{n}": 0.0 for n in indicator_names},
                **{f"uniq_{n}": 1.0 for n in indicator_names}}
    raise ValueError(f"Unknown LCA mode: {mode}")


def _run_em_lca(n: int, K: int, rng: np.random.Generator, max_iter: int, tol: float,
                 mode: str, indicator_names: list, Xind: np.ndarray,
                 iv_names: Optional[list], Xreg: Optional[np.ndarray], y: Optional[np.ndarray],
                 fixed_class: Optional[np.ndarray], equality_constraints: Optional[list]) -> tuple:
    """
    Single EM run for general LCA/finite mixture models — mirrors
    fimix.py's ``_run_em`` E-/M-step structure, generalized across the three
    supported modes and semi-supervised (fixed-label) rows.

    Returns (tau, seg_params, pi_k, log_lik).
    """
    raw = rng.dirichlet(np.ones(K), size=n)
    tau = raw / raw.sum(axis=1, keepdims=True)
    if fixed_class is not None:
        tau = _apply_fixed_labels(tau, fixed_class, K)
    pi_k = tau.mean(axis=0)
    seg_params = [{} for _ in range(K)]
    log_lik = -np.inf

    for _iter in range(max_iter):

        # ── M-step ───────────────────────────────────────────────────────
        pi_k = tau.mean(axis=0)
        class_weight_sums = tau.sum(axis=0)
        for k in range(K):
            w = tau[:, k]
            if w.sum() < 1e-6:
                seg_params[k] = _degenerate_params(mode, indicator_names, iv_names)
                continue
            if mode == "segmentation":
                seg_params[k] = _mstep_segmentation(Xind, w, indicator_names)
            elif mode == "mixture_regression":
                coefs, _, sigma2 = _ols_paths(Xreg, y, w)
                seg_params[k] = {name: float(c) for name, c in zip(iv_names, coefs)}
                seg_params[k]["sigma2"] = float(sigma2)
            elif mode == "mixture_factor":
                seg_params[k] = _mstep_factor(Xind, w, indicator_names)
            else:
                raise ValueError(f"Unknown LCA mode: {mode}")

        if equality_constraints:
            _apply_equality_constraints(seg_params, class_weight_sums, equality_constraints)

        # ── E-step ───────────────────────────────────────────────────────
        log_p = np.zeros((n, K))
        for k in range(K):
            lp = np.full(n, np.log(max(pi_k[k], 1e-300)))
            if mode == "segmentation":
                lp = lp + _loglik_segmentation(Xind, seg_params[k], indicator_names)
            elif mode == "mixture_regression":
                coefs  = np.array([seg_params[k][name] for name in iv_names])
                sigma2 = max(seg_params[k]["sigma2"], 1e-14)
                resid  = y - Xreg @ coefs
                lp = lp - 0.5 * np.log(2 * np.pi * sigma2) - 0.5 * resid ** 2 / sigma2
            elif mode == "mixture_factor":
                lp = lp + _loglik_factor(Xind, seg_params[k], indicator_names)
            log_p[:, k] = lp

        log_p_max = log_p.max(axis=1, keepdims=True)
        p_stable  = np.exp(log_p - log_p_max)
        row_sums  = p_stable.sum(axis=1, keepdims=True)
        tau_new   = p_stable / np.maximum(row_sums, 1e-300)

        if fixed_class is not None:
            tau_new = _apply_fixed_labels(tau_new, fixed_class, K)

        new_ll = float(
            np.sum(log_p_max.ravel() + np.log(np.maximum(row_sums.ravel(), 1e-300)))
        )
        delta   = abs(new_ll - log_lik)
        tau     = tau_new
        log_lik = new_ll

        if delta < tol:
            break

    return tau, seg_params, pi_k, log_lik


# ── public API ───────────────────────────────────────────────────────────────

def run_lca(
    df: pd.DataFrame,
    indicator_cols: list,
    k_range: tuple = (2, 6),
    mode: LCAMode = "segmentation",
    dv_col: Optional[str] = None,
    known_class_col: Optional[str] = None,
    equality_constraints: Optional[list] = None,
    n_starts: int = 10,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> LCAResult:
    """
    Run general latent class / finite mixture analysis for K = k_range[0] .. k_range[1].

    Parameters
    ----------
    df                    : Tidy observation-level DataFrame.
    indicator_cols        : Columns defining class membership.
                             - "segmentation":       the class-profile indicators.
                             - "mixture_regression":  the IV columns (dv_col is the DV).
                             - "mixture_factor":      the single-factor indicators.
    k_range               : (k_min, k_max) inclusive range of classes to test.
    mode                  : "segmentation" | "mixture_regression" | "mixture_factor".
    dv_col                : Dependent variable column — required for "mixture_regression".
    known_class_col       : Optional column of known class labels (0..K-1) for
                             semi-supervised seeding (A14). Rows with a non-null
                             value have their responsibility fixed to that class
                             for every E-step; null rows are classified by EM.
    equality_constraints  : Optional list of parameter names (in the mode's
                             M-step naming convention) to estimate as equal
                             across all classes (A15).
    n_starts, max_iter, tol, seed, log_fn : as in ``run_fimix``.

    Returns
    -------
    LCAResult with a fit-table row per converged K, per-case posterior
    membership, per-class (or constraint-pooled) parameters, and a
    recommended_k chosen via the same entropy-gated CAIC rule used by
    FIMIX-PLS (``app.fimix.recommend_k``).
    """
    k_min, k_max = k_range
    _emit(log_fn, "step", f"LCA[{mode}]: K={k_min}..{k_max}, {n_starts} starts, {max_iter} iter")

    missing = [c for c in indicator_cols if c not in df.columns]
    if missing:
        raise ValueError(f"LCA: indicator column(s) not found in data: {missing}")
    if mode == "mixture_regression":
        if not dv_col:
            raise ValueError("LCA: dv_col is required for mode='mixture_regression'.")
        if dv_col not in df.columns:
            raise ValueError(f"LCA: dv_col '{dv_col}' not found in data.")

    n = len(df)
    rng = np.random.default_rng(seed)
    warnings: list = []

    Xind = df[indicator_cols].values.astype(float)
    iv_names = indicator_cols if mode == "mixture_regression" else None
    Xreg = Xind if mode == "mixture_regression" else None
    y = df[dv_col].values.astype(float) if mode == "mixture_regression" else None

    fit_table: list[LCAFitRow] = []
    class_sizes: dict = {}
    per_case_membership: dict = {}
    parameters: dict = {}

    for K in range(k_min, k_max + 1):
        _emit(log_fn, "step", f"  LCA K={K}")

        fixed_class = None
        if known_class_col is not None:
            # A validation failure (bad labels) is a fatal input error, not a
            # per-K convergence issue — let it propagate rather than being
            # swallowed into warnings for every K in the range.
            fixed_class = _validate_known_class_col(df, known_class_col, K)

        best_ll = -np.inf
        best_tau = best_seg_params = best_pi = None

        for _s in range(n_starts):
            try:
                tau, seg_params, pi_k, ll = _run_em_lca(
                    n, K, rng, max_iter, tol, mode, indicator_cols, Xind,
                    iv_names, Xreg, y, fixed_class, equality_constraints,
                )
                if ll > best_ll:
                    best_ll, best_tau, best_seg_params, best_pi = ll, tau, seg_params, pi_k
            except Exception as exc:
                warnings.append(f"LCA K={K} start failed: {exc}")

        if best_tau is None:
            warnings.append(f"LCA K={K}: all starts failed — skipped")
            continue

        # ── free-parameter count (equality constraints reduce it) ─────────
        base_per_class = len(best_seg_params[0]) if best_seg_params else 0
        n_constrained = len(
            set(equality_constraints or []) & set(best_seg_params[0].keys())
        ) if best_seg_params else 0
        n_params = K * (base_per_class - n_constrained) + n_constrained + (K - 1)

        aic  = round(-2 * best_ll + 2 * n_params, 4)
        bic  = round(-2 * best_ll + np.log(n) * n_params, 4)
        caic = round(-2 * best_ll + (np.log(n) + 1) * n_params, 4)
        r_e  = _relative_entropy(best_tau)   # reused verbatim from fimix.py

        fit_table.append(LCAFitRow(
            k=K, log_likelihood=round(best_ll, 4), aic=aic, bic=bic, caic=caic,
            relative_entropy=r_e,
        ))

        hard_assign = np.argmax(best_tau, axis=1)
        sizes = {c: int(np.sum(hard_assign == c)) for c in range(K)}
        class_sizes[K] = sizes
        per_case_membership[K] = best_tau.round(6).tolist()

        class_params: list[LCAClassParameters] = []
        for k in range(K):
            class_params.append(LCAClassParameters(
                class_id=k,
                size=sizes[k],
                proportion=round(float(best_pi[k]), 6),
                parameters={name: round(float(v), 6) for name, v in best_seg_params[k].items()},
            ))
        parameters[K] = class_params

    if not fit_table:
        raise ValueError("LCA: no solutions converged for any K in the requested range.")

    recommended_k, rec_warnings = recommend_k(fit_table, label="LCA")
    warnings.extend(rec_warnings)

    _emit(log_fn, "ok", f"LCA[{mode}] complete — recommended K={recommended_k}")

    return LCAResult(
        mode=mode,
        n_obs=n,
        indicator_cols=list(indicator_cols),
        class_sizes=class_sizes,
        fit_table=fit_table,
        recommended_k=recommended_k,
        per_case_membership=per_case_membership,
        parameters=parameters,
        equality_constraints=list(equality_constraints or []),
        known_class_col=known_class_col,
        warnings=warnings,
    )
