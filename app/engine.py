"""
NAVAL-SEM Statistical Engine
Wraps semopy with clean interfaces for:
  - CB-SEM / WLS  (via semopy.Model)
  - PLS-SEM       (via semopy.PLS, falls back to semopy.Model if PLS unavailable)
  - Bootstrapping
  - HTMT (Heterotrait-Monotrait ratio)
  - Code Export   (R/lavaan, Python/semopy, .lav)
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger("naval_sem.engine")


from app.engine_utils import _emit, _safe_float, _p_to_sig, _build_composites
from app.parser import parse_lavaan, build_semopy_syntax
from app.schemas import (
    ModelResult, PathParameter, FitIndices,
    BootstrapResult, HTMTResult, HTMTEntry,
    VIFEntry, F2Entry, IndirectEffect, IndirectResult, OuterWeightEntry,
    Q2Entry, PLSPredictEntry, CVPATResult, CMBMarkerResult, PredictResult,
    ModelSummary, StructuralPathSummary, ConstructValiditySummary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
# _emit, _safe_float, _p_to_sig, _build_composites live in engine_utils.py


def _fit_verdict(fit: FitIndices) -> FitIndices:
    if fit.cfi is not None:
        # Thresholds are CB-SEM (Hu & Bentler 1999).
        # For PLS, cfi here is ULS-based; these verdicts are informational only.
        fit.cfi_acceptable = fit.cfi >= 0.90
        fit.cfi_good = fit.cfi >= 0.95
    if fit.tli is not None:
        fit.tli_acceptable = fit.tli >= 0.90
        fit.tli_good = fit.tli >= 0.95
    if fit.rmsea is not None:
        # Thresholds are CB-SEM (MacCallum et al. 1996).
        # For PLS, rmsea here is ULS-based; these verdicts are informational only.
        fit.rmsea_acceptable = fit.rmsea <= 0.08
        fit.rmsea_good = fit.rmsea <= 0.06
    if fit.srmr is not None:
        fit.srmr_good = fit.srmr <= 0.08
    # Fornell-Larcker pass verdict: already set during computation,
    # but re-derive here in case fit was mutated after the fact.
    if fit.fornell_larcker is not None and fit.ave is not None:
        lvs = list(fit.fornell_larcker.keys())
        passed = True
        for lv_a in lvs:
            sqrt_ave_a = fit.ave.get(lv_a)
            if sqrt_ave_a is None:
                passed = False
                break
            sqrt_ave_a = sqrt_ave_a ** 0.5
            for lv_b, r_ab in fit.fornell_larcker[lv_a].items():
                if lv_a == lv_b:
                    continue  # diagonal entry is √AVE itself, skip self-comparison
                if sqrt_ave_a <= abs(r_ab):
                    passed = False
                    break
            if not passed:
                break
        fit.fornell_larcker_pass = passed
    return fit


# ── Measurement validity helpers ──────────────────────────────────────────────

def _extract_loadings(
    params_df: pd.DataFrame,
    measurement: dict[str, list[str]],
    df: Optional[pd.DataFrame] = None,
) -> dict[str, list[float]]:
    """
    Return {lv_name: [standardized_loading, ...]} ready for AVE / CR formulas.

    Two-stage strategy
    ------------------
    Stage 1 – semopy lookup:
      Build a (left, right) → estimate index from inspect() output, op-agnostic.
      This handles both semopy column naming styles (lval/rval and lhs/rhs).

    Stage 2 – auto-standardize if needed:
      semopy's ML estimator returns *unstandardized* loadings (one per factor is
      fixed to 1.0 for identification; others can exceed 1).  AVE and CR require
      standardized loadings (|λ| ≤ 1).  When any |loading| > 1 we re-derive all
      loadings as corr(indicator, construct_composite), where the composite is the
      unweighted mean of that LV's indicators.  This is data-driven, always
      bounded in [−1, 1], and consistent with how most PLS-SEM software reports
      loadings.

    Falls back to op == '=~' scan for pure path models (no measurement dict).
    """
    if not measurement:
        loadings: dict[str, list[float]] = {}
        for _, row in params_df.iterrows():
            if str(row.get("op", "")) != "=~":
                continue
            lv  = str(row.get("lval", row.get("lhs", "")))
            est = _safe_float(row.get("Estimate", row.get("estimate", None)))
            if est is not None:
                loadings.setdefault(lv, []).append(est)
        return loadings

    # ── Stage 1: build estimate lookup ───────────────────────────────────────
    est_col = "Estimate" if "Estimate" in params_df.columns else "estimate"
    pair_to_est: dict[tuple[str, str], float] = {}
    for _, row in params_df.iterrows():
        left  = str(row.get("lval", row.get("lhs", "")))
        right = str(row.get("rval", row.get("rhs", "")))
        est   = _safe_float(row.get(est_col, None))
        if est is not None:
            pair_to_est[(left, right)] = est
            pair_to_est[(right, left)] = est

    loadings: dict[str, list[float]] = {}
    for lv, indicators in measurement.items():
        for ind in indicators:
            est = pair_to_est.get((lv, ind)) or pair_to_est.get((ind, lv))
            if est is not None:
                loadings.setdefault(lv, []).append(est)

    # ── Stage 2: proper CFA standardization — λ* = λ√φ / √(λ²φ + θ) ──────────
    # semopy ML fixes one indicator per factor to 1.0 for identification, so
    # unstandardised loadings routinely exceed 1.  We re-express everything on a
    # unit-variance scale using the latent variance (φ_jj) and error variance (θ_ii)
    # that are already in params_df as diagonal ~~ rows.
    # Falls back to sample-variance denominator when θ is absent, and to composite
    # correlation (the old approach) only when φ itself is unavailable.
    needs_std = any(abs(l) > 1.0 for lams in loadings.values() for l in lams)
    if needs_std:
        all_lvs  = set(measurement.keys())
        all_inds = {ind for inds in measurement.values() for ind in inds}

        # Harvest φ (latent variances) and θ (error variances) from diagonal ~~ rows
        phi:   dict[str, float] = {}
        theta: dict[str, float] = {}
        for _, row in params_df.iterrows():
            op    = str(row.get("op", ""))
            left  = str(row.get("lval", row.get("lhs", "")))
            right = str(row.get("rval", row.get("rhs", "")))
            val   = _safe_float(row.get(est_col))
            if op == "~~" and left == right and val is not None:
                if left in all_lvs:
                    phi[left] = val
                elif left in all_inds:
                    theta[left] = val

        std_loadings: dict[str, list[float]] = {}
        for lv, indicators in measurement.items():
            phi_jj = phi.get(lv)
            lam_std = []

            if phi_jj is not None and phi_jj > 0:
                for ind in indicators:
                    lam_raw = pair_to_est.get((lv, ind)) or pair_to_est.get((ind, lv))
                    if lam_raw is None:
                        continue
                    theta_ii = theta.get(ind)
                    if theta_ii is not None:
                        # Full model-based: λ* = λ√φ / √(λ²φ + θ)
                        var_x = lam_raw ** 2 * phi_jj + theta_ii
                    elif df is not None and ind in df.columns:
                        # φ known but θ absent — use sample variance as denominator
                        var_x = float(df[ind].var())
                    else:
                        continue
                    if var_x > 0:
                        val = _safe_float(lam_raw * np.sqrt(phi_jj) / np.sqrt(var_x))
                        if val is not None:
                            lam_std.append(val)

            if lam_std:
                std_loadings[lv] = lam_std
            elif df is not None:
                # φ unavailable — fall back to composite-correlation approximation
                cols = [c for c in indicators if c in df.columns]
                if cols:
                    composite = df[cols].mean(axis=1)
                    comp_std  = composite.std()
                    if comp_std > 0:
                        fallback = []
                        for ind in indicators:
                            if ind not in df.columns:
                                continue
                            r = float(df[ind].corr(composite))
                            v = _safe_float(r)
                            if v is not None:
                                fallback.append(v)
                        if fallback:
                            std_loadings[lv] = fallback

            if lv not in std_loadings:
                std_loadings[lv] = loadings.get(lv, [])

        return std_loadings

    return loadings


def _compute_srmr(
    sem_model,
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    parsed: dict,
) -> Optional[float]:
    """
    Compute SRMR (Standardized Root Mean Square Residual) manually.
    semopy's calc_stats does not include SRMR, so we build it here.

    SRMR = sqrt( (2 / p(p+1)) * Σ_{i≥j} ((s_ij − σ_ij) / sqrt(s_ii * s_jj))² )

    where s_ij  = sample covariance, σ_ij = model-implied covariance.

    Strategy
    --------
    1. Try semopy's internal implied-covariance matrix (attribute scan).
    2. Reconstruct Σ from CFA parameter estimates:
       Σ = Λ Φ Λᵀ + Θ
       where Λ = loading matrix, Φ = latent (co)variances, Θ = error variances.
       If Θ is missing (semopy sometimes omits it), estimate from data residuals.
    """
    observed_vars = parsed.get("observed_vars", [])
    latent_vars   = parsed.get("latent_vars", [])
    measurement   = parsed.get("measurement", {})

    obs = [v for v in observed_vars if v in df.columns]
    p   = len(obs)
    if p < 2:
        return None

    obs_idx = {v: i for i, v in enumerate(obs)}
    lat_idx = {v: i for i, v in enumerate(latent_vars)}
    q = len(latent_vars)

    est_col = "Estimate" if "Estimate" in params_df.columns else "estimate"

    # ── Step 1: try semopy's implied covariance attribute ─────────────────────
    # ── Resolve the definitive observed-variable order from the model ────────
    # semopy's mx_cov rows/cols follow m.vars['observed'], NOT the order that
    # parse_lavaan builds from a Python set (which is non-deterministic).
    # Using the wrong order mis-aligns S and Sigma, inflating SRMR dramatically.
    model_obs_order: Optional[list] = None
    if hasattr(sem_model, "vars") and isinstance(sem_model.vars, dict):
        model_obs_order = sem_model.vars.get("observed")
        if model_obs_order:
            # Restrict to vars actually present in df, preserving model order
            model_obs_order = [v for v in model_obs_order if v in df.columns]

    # If model gives us an order, rebuild obs/p/obs_idx with it
    if model_obs_order:
        obs      = model_obs_order
        p        = len(obs)
        obs_idx  = {v: i for i, v in enumerate(obs)}

    Sigma = None
    for attr in ("mx_cov", "sigma", "implied_cov", "cov_implied"):
        if hasattr(sem_model, attr):
            try:
                raw = getattr(sem_model, attr)
                if hasattr(raw, "values"):
                    raw = raw.values
                arr = np.array(raw, dtype=float)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1] == p:
                    Sigma = arr
                    break
            except Exception as _e:  # B110
                logger.debug("Non-critical exception suppressed: %s", _e)
                pass

    # ── Step 2: reconstruct from CFA parameters ───────────────────────────────
    if Sigma is None and q > 0:
        try:
            # Build (left, right) → estimate lookup (bidirectional)
            param_lut: dict[tuple[str, str], float] = {}
            for _, row in params_df.iterrows():
                left  = str(row.get("lval", row.get("lhs", "")))
                right = str(row.get("rval", row.get("rhs", "")))
                val   = _safe_float(row.get(est_col))
                if val is not None:
                    param_lut[(left, right)] = val
                    param_lut[(right, left)] = val

            # Λ: loading matrix (p × q)
            Lam = np.zeros((p, q))
            for lv, indicators in measurement.items():
                if lv not in lat_idx:
                    continue
                j = lat_idx[lv]
                for ind in indicators:
                    if ind not in obs_idx:
                        continue
                    i = obs_idx[ind]
                    val = param_lut.get((lv, ind))
                    if val is not None:
                        Lam[i, j] = val

            # Φ: latent (co)variance matrix (q × q); default = identity
            Phi = np.eye(q)
            for j1, lv1 in enumerate(latent_vars):
                for j2, lv2 in enumerate(latent_vars):
                    val = param_lut.get((lv1, lv2))
                    if val is not None:
                        Phi[j1, j2] = val

            # Θ: error variance matrix (p × p diagonal)
            Theta = np.zeros((p, p))
            for i, obs_var in enumerate(obs):
                val = param_lut.get((obs_var, obs_var))
                if val is not None:
                    Theta[i, i] = val

            Sigma_no_theta = Lam @ Phi @ Lam.T

            # If Theta is empty, estimate residual variances from data
            if np.all(Theta == 0):
                S_diag = df[obs].var(numeric_only=True).values
                implied_diag = np.diag(Sigma_no_theta)
                for i in range(p):
                    Theta[i, i] = max(S_diag[i] - implied_diag[i], 0.0)

            Sigma = Sigma_no_theta + Theta
        except Exception as _e:  # B110
            logger.debug("Non-critical exception suppressed: %s", _e)
            pass

    if Sigma is None:
        return None

    # ── Step 3: compute SRMR ──────────────────────────────────────────────────
    try:
        S = df[obs].cov(numeric_only=True).values
        if S.shape != Sigma.shape:
            return None
        total = 0.0
        count = 0
        for i in range(p):
            for j in range(i + 1):   # lower triangle including diagonal
                denom = np.sqrt(abs(S[i, i] * S[j, j]))
                if denom > 0:
                    total += ((S[i, j] - Sigma[i, j]) / denom) ** 2
                    count += 1
        if count == 0:
            return None
        return _safe_float(np.sqrt(2.0 * total / (p * (p + 1))))
    except Exception as _e:
        return None



def _compute_std_estimates(
    params_df: pd.DataFrame,
    measurement: dict[str, list[str]],
) -> dict[tuple[str, str, str], float]:
    """
    Compute standardised estimates (std.all equivalent) for every row in params_df.

    Returns a dict keyed by (lhs, op, rhs) → std_estimate.

    Standardisation rules
    ---------------------
    =~  measurement rows  :  λ* = λ · √φ_jj / √(λ²·φ_jj + θ_ii)
                             where φ_jj = latent variance, θ_ii = error variance.
    ~~  latent covariances :  r  = cov_ab / √(var_a · var_b)        (→ correlation)
    ~~  diagonal / error   :  carried through unchanged (variance already on std scale)
    ~   structural paths   :  left unchanged (require factor scores for full std.all;
                             callers may override with bootstrap-based SE later)
    """
    est_col = "Estimate" if "Estimate" in params_df.columns else "estimate"
    all_lvs  = set(measurement.keys())
    all_inds = {ind for inds in measurement.values() for ind in inds}

    # ── Harvest raw estimates (bidirectional for easy lookup) ─────────────────
    pair_to_est: dict[tuple[str, str], float] = {}
    for _, row in params_df.iterrows():
        left  = str(row.get("lval", row.get("lhs", "")))
        right = str(row.get("rval", row.get("rhs", "")))
        val   = _safe_float(row.get(est_col))
        if val is not None:
            pair_to_est[(left, right)] = val
            pair_to_est[(right, left)] = val

    # ── Latent variances φ and error variances θ from diagonal ~~ rows ────────
    phi:   dict[str, float] = {}   # {lv:  φ_jj}
    theta: dict[str, float] = {}   # {ind: θ_ii}
    for _, row in params_df.iterrows():
        op    = str(row.get("op", ""))
        left  = str(row.get("lval", row.get("lhs", "")))
        right = str(row.get("rval", row.get("rhs", "")))
        val   = _safe_float(row.get(est_col))
        if op == "~~" and left == right and val is not None:
            if left in all_lvs:
                phi[left] = val
            else:
                theta[left] = val   # indicator error variance

    # ── Build indicator → owning LV map ──────────────────────────────────────
    ind_to_lv: dict[str, str] = {}
    for lv, indicators in measurement.items():
        for ind in indicators:
            ind_to_lv[ind] = lv

    result: dict[tuple[str, str, str], float] = {}

    for _, row in params_df.iterrows():
        op    = str(row.get("op", ""))
        left  = str(row.get("lval", row.get("lhs", "")))
        right = str(row.get("rval", row.get("rhs", "")))
        val   = _safe_float(row.get(est_col))
        if val is None:
            continue

        if op == "=~":
            # left = LV, right = indicator
            lv  = left
            ind = right
            phi_jj   = phi.get(lv)
            theta_ii = theta.get(ind)
            if phi_jj is not None and phi_jj > 0 and theta_ii is not None:
                var_x = val ** 2 * phi_jj + theta_ii
                if var_x > 0:
                    std = _safe_float(val * np.sqrt(phi_jj) / np.sqrt(var_x))
                    if std is not None:
                        result[(left, op, right)] = std

        elif op == "~~":
            if left == right:
                # Diagonal: standardised variance = 1 for LVs (by definition in
                # std.all); for indicators it's the proportion of variance that is
                # error = θ / Var(x).  We store the raw value; callers can ignore.
                result[(left, op, right)] = val
            else:
                # Off-diagonal: convert covariance → correlation
                var_a = phi.get(left)  or phi.get(right)  or \
                        theta.get(left) or theta.get(right)
                # Both sides must be LVs for the phi lookup to be valid
                phi_a = phi.get(left)
                phi_b = phi.get(right)
                if phi_a is not None and phi_b is not None and phi_a > 0 and phi_b > 0:
                    corr = _safe_float(val / np.sqrt(phi_a * phi_b))
                    if corr is not None:
                        result[(left, op, right)] = corr
                        result[(right, op, left)] = corr   # symmetric

        # ~ structural: skip — standardisation requires factor scores
        # (bootstrap SE back-fill handles significance; std.all betas are out of scope here)

    return result


def _compute_ave(loadings: dict[str, list[float]]) -> dict[str, float]:
    """AVE = Σλ² / n  for each latent variable."""
    ave = {}
    for lv, lambdas in loadings.items():
        if not lambdas:
            continue
        lam = np.array(lambdas, dtype=float)
        val = _safe_float(np.sum(lam ** 2) / len(lam))
        if val is not None:
            ave[lv] = val
    return ave


def _compute_composite_reliability(loadings: dict[str, list[float]]) -> dict[str, float]:
    """ρc = (Σλ)² / ((Σλ)² + Σ(1 - λ²))"""
    cr = {}
    for lv, lambdas in loadings.items():
        if not lambdas:
            continue
        lam = np.array(lambdas, dtype=float)
        sum_lam = np.sum(lam)
        sum_err = np.sum(1.0 - lam ** 2)
        denom = sum_lam ** 2 + sum_err
        val = _safe_float(sum_lam ** 2 / denom if denom > 0 else None)
        if val is not None:
            cr[lv] = val
    return cr


def _compute_cronbach_alpha(
    df: pd.DataFrame,
    measurement: dict[str, list[str]],
) -> dict[str, float]:
    """
    Standard Cronbach α using the indicator covariance matrix.
    α = (k / (k-1)) * (1 - Σvar_i / var_total)
    Only computed when k >= 2 and var_total > 0.
    Result is clamped to [0, 1] — values above 1.0 indicate highly
    inter-correlated items (common with synthetic data).
    """
    alpha = {}
    for lv, indicators in measurement.items():
        cols = [c for c in indicators if c in df.columns]
        if len(cols) < 2:
            continue
        try:
            cov = df[cols].cov(numeric_only=True)
            k = len(cols)
            var_total = float(cov.values.sum())
            var_items = float(np.trace(cov.values))
            if var_total <= 0:
                continue
            raw = (k / (k - 1)) * (1.0 - var_items / var_total)
            val = _safe_float(min(max(raw, 0.0), 1.0))
            if val is not None:
                alpha[lv] = val
        except Exception as _e:  # B112
            logger.debug("Non-critical exception suppressed: %s", _e)
            continue
    return alpha


def _compute_fornell_larcker(
    ave: dict[str, float],
    df: pd.DataFrame,
    measurement: dict[str, list[str]],
) -> tuple[dict[str, dict[str, float]], bool]:
    """
    Build the Fornell-Larcker matrix.
    Diagonal = √AVE for that LV.
    Off-diagonal = inter-construct correlation (average correlation between
    indicators of different LVs, matching the HTMT approach).

    Returns (matrix_dict, all_pass).
    """
    lvs = list(ave.keys())
    corr = df.corr(numeric_only=True)

    def mean_cross_corr(inds_a: list[str], inds_b: list[str]) -> Optional[float]:
        vals = []
        for a in inds_a:
            for b in inds_b:
                if a in corr.columns and b in corr.columns:
                    vals.append(corr.loc[a, b])
        return float(np.mean(vals)) if vals else None

    matrix: dict[str, dict[str, float]] = {}
    all_pass = True

    for lv_a in lvs:
        matrix[lv_a] = {}
        sqrt_ave_a = _safe_float(ave[lv_a] ** 0.5)
        # Diagonal: √AVE
        matrix[lv_a][lv_a] = sqrt_ave_a if sqrt_ave_a is not None else 9999.0

        for lv_b in lvs:
            if lv_b == lv_a:
                continue
            inds_a = measurement.get(lv_a, [])
            inds_b = measurement.get(lv_b, [])
            r_ab = mean_cross_corr(inds_a, inds_b)
            r_val = _safe_float(r_ab) if r_ab is not None else None
            matrix[lv_a][lv_b] = r_val if r_val is not None else 9999.0

            if sqrt_ave_a is not None and r_val is not None:
                if sqrt_ave_a <= abs(r_val):
                    all_pass = False
            else:
                all_pass = False

    return matrix, all_pass


# ── PLS global fit indices ───────────────────────────────────────────────────

def _compute_pls_global_fit(
    df: pd.DataFrame,
    pls_result,
    parsed: dict,
) -> tuple[Optional[float], Optional[float]]:
    """
    Approximate CFI and RMSEA for PLS-SEM using the ULS discrepancy.

    Uses the Unweighted Least Squares discrepancy F_ULS = 0.5·tr[(S−Σ)²]
    rather than the ML discrepancy, which is required for CB-SEM but
    inappropriate for PLS (PLS does not minimise F_ML; applying it produces
    systematically inflated chi-square and poor CFI/RMSEA vs CB-SEM benchmarks).

    The Hu & Bentler (1999) cutoffs CFI≥0.90 / RMSEA<0.08 DO NOT APPLY to
    these values.  Use SRMR (computed in pls.py) as the primary global fit
    criterion for PLS-SEM.  These CFI/RMSEA values are supplementary only.

    Degrees of freedom (covariance-structure approximation)
    -------------------------------------------------------
      n_free_params  = p (outer loadings) + q*(q-1)/2 (unique LV correlations)
      df_model       = p*(p-1)/2 − n_free_params        (unique off-diagonals)
      df_null        = p*(p-1)/2                          (independence model)

    Reference: Henseler, Ringle & Sarstedt (2014) JAMS 43(1), 115–135.

    Returns (cfi, rmsea) in [0, 1], or (None, None) on any numeric error.
    """
    try:
        measurement = parsed.get("measurement", {})
        if not measurement:
            return None, None

        # ── Gather indicators in a stable order ───────────────────────────
        all_indicators: list[str] = []
        lv_for_ind: dict[str, str] = {}
        for lv, inds in measurement.items():
            for ind in inds:
                if ind not in lv_for_ind:
                    all_indicators.append(ind)
                    lv_for_ind[ind] = lv

        obs = [ind for ind in all_indicators if ind in df.columns]
        p   = len(obs)
        n   = len(df)
        if p < 2 or n <= p:
            return None, None

        # ── Loading lookup ────────────────────────────────────────────────
        lam: dict[str, float] = {}
        for lv, ind_map in pls_result.outer_loadings.items():
            for ind, loading in ind_map.items():
                lam[ind] = float(loading)

        # ── LV inter-correlations from actual PLS scores ──────────────────
        lvs     = [lv for lv in measurement if lv in pls_result.outer_loadings]
        q       = len(lvs)
        lv_idx  = {lv: i for i, lv in enumerate(lvs)}
        scores_df  = pls_result.scores
        scores_arr = (
            scores_df[lvs].values
            if hasattr(scores_df, "columns")
            else scores_df
        )
        phi = np.corrcoef(scores_arr.T) if q > 1 else np.array([[1.0]])
        phi = np.clip(phi, -1.0, 1.0)

        # ── Sample correlation matrix S ───────────────────────────────────
        S = df[obs].corr(numeric_only=True).values.astype(float)

        # ── Model-implied correlation matrix Σ ────────────────────────────
        Sigma = np.zeros((p, p))
        for i, ind_i in enumerate(obs):
            lv_i  = lv_for_ind.get(ind_i)
            lam_i = lam.get(ind_i, 0.0)
            for j, ind_j in enumerate(obs):
                lv_j  = lv_for_ind.get(ind_j)
                lam_j = lam.get(ind_j, 0.0)
                if i == j:
                    Sigma[i, j] = 1.0
                elif lv_i is None or lv_j is None:
                    Sigma[i, j] = S[i, j]          # passthrough
                elif lv_i == lv_j:
                    Sigma[i, j] = lam_i * lam_j    # within-block
                else:
                    li = lv_idx.get(lv_i, 0)
                    lj = lv_idx.get(lv_j, 0)
                    Sigma[i, j] = lam_i * phi[li, lj] * lam_j  # cross-block

        # Regularise for positive-definiteness
        eps   = 1e-6
        S_r   = S     + eps * np.eye(p)
        Sig_r = Sigma + eps * np.eye(p)

        # ── ULS discrepancy: F_ULS = 0.5 · tr[(S − Σ)²]  ─────────────────
        # Preferred over ML discrepancy for PLS: no normality / ML assumption.
        # (Henseler, Ringle & Sarstedt 2014; Hair et al. 2022 ch. 4)
        residual   = S_r - Sig_r
        F_model    = 0.5 * float(np.einsum("ij,ij->", residual, residual))
        chi2_model = (n - 1) * F_model

        # ── Degrees of freedom ────────────────────────────────────────────
        n_free   = p + q * (q - 1) // 2   # loadings + unique LV correlations
        df_model = max(1, p * (p - 1) // 2 - n_free)

        # ── Null (independence): Σ_null = I  →  residual = S − I (off-diags only)
        off_diag = S_r.copy()
        np.fill_diagonal(off_diag, 0.0)
        F_null    = 0.5 * float(np.einsum("ij,ij->", off_diag, off_diag))
        chi2_null = (n - 1) * F_null
        df_null   = p * (p - 1) // 2       # all off-diagonals constrained = 0

        # ── CFI (Bentler 1990 formula, ULS-based ncp) ─────────────────────
        ncp_model = max(0.0, chi2_model - df_model)
        ncp_null  = max(1e-12, chi2_null  - df_null)
        cfi       = round(max(0.0, min(1.0, 1.0 - ncp_model / ncp_null)), 6)

        # ── RMSEA (Steiger 1990 formula, ULS-based) ───────────────────────
        rmsea_sq = max(0.0, (chi2_model - df_model) / (df_model * (n - 1)))
        rmsea    = round(min(1.0, float(np.sqrt(rmsea_sq))), 6)

        return cfi, rmsea

    except Exception:
        return None, None


# ── Main fit function ─────────────────────────────────────────────────────────

def fit_model(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
    bootstrap_n: int = 0,
    log_fn: Optional[Callable] = None,
) -> ModelResult:
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed. Run: pip install semopy")

    _emit(log_fn, "step", "Parsing lavaan syntax")
    parsed = parse_lavaan(model_syntax)
    syntax = build_semopy_syntax(parsed)
    warnings = []

    latent_set = set(parsed.get("latent_vars", []))
    for cov in parsed.get("covariances", []):
        lhs, rhs = cov["lhs"], cov["rhs"]
        if lhs not in latent_set or rhs not in latent_set:
            msg = f"Covariance '{lhs} ~~ {rhs}' uses observed indicators. semopy may fail to converge or produce NaN estimates."
            _emit(log_fn, "warn", msg)
            warnings.append(msg)

    n_lv = len(parsed["latent_vars"])
    n_obs_vars = len(parsed["observed_vars"])
    n_struct = len(parsed["structural"])
    _emit(log_fn, "info", f"Model structure: {n_lv} latent vars · {n_obs_vars} indicators · {n_struct} structural paths")
    _emit(log_fn, "info", f"Data: {len(df)} observations · {len(df.columns)} columns")

    missing_cols = [v for v in parsed["observed_vars"] if v not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found in data: {missing_cols}. "
            f"Available: {df.columns.tolist()}"
        )

    _emit(log_fn, "info", "Column check passed — all indicators found in data")
    estimator = "ML"
    use_pls = (algorithm == "pls")
    if algorithm == "wls":
        estimator = "WLS"

    _emit(log_fn, "step", f"Initializing {algorithm.upper()} estimator")

    # ── PLS-SEM branch ────────────────────────────────────────────────────────
    if use_pls:
        from app.pls import PLSEstimator, pls_loadings_to_list
        try:
            pls_result = PLSEstimator().fit(df, parsed)
        except Exception as e:
            raise ValueError(f"PLS-SEM did not converge: {e}")

        warnings.extend(pls_result.warnings)
        algo_label = "PLS-SEM"

        _emit(log_fn, "step", "Extracting PLS parameter estimates")
        parameters: list[PathParameter] = []

        # Outer loadings (=~ edges)
        for lv, ind_map in pls_result.outer_loadings.items():
            for ind, loading in ind_map.items():
                parameters.append(PathParameter(
                    lhs=lv, op="=~", rhs=ind,
                    estimate=round(loading, 6),
                    std_error=0.0,
                    z_value=0.0,
                    p_value=1.0,          # filled by bootstrap back-fill below
                    ci_lower=None,
                    ci_upper=None,
                    significant=False,    # filled by bootstrap back-fill below
                ))

        # Structural paths (~ edges)
        for lhs, rhs_map in pls_result.path_coefficients.items():
            for rhs, coef in rhs_map.items():
                parameters.append(PathParameter(
                    lhs=lhs, op="~", rhs=rhs,
                    estimate=round(coef, 6),
                    std_error=0.0,
                    z_value=0.0,
                    p_value=1.0,
                    ci_lower=None,
                    ci_upper=None,
                    significant=False,
                ))

        _emit(log_fn, "step", "Computing PLS fit indices (SRMR · AVE · CR)")
        fit = FitIndices()

        # PLS-SEM: approximate global fit via ML discrepancy function
        # (Lohmöller 1989; Henseler et al. 2014; Bentler / Steiger formulae)
        _pls_cfi, _pls_rmsea = _compute_pls_global_fit(df, pls_result, parsed)
        fit.cfi        = _pls_cfi
        fit.rmsea      = _pls_rmsea
        fit.chi_square = None
        fit.df         = None
        fit.p_value    = None
        fit.aic        = None
        fit.bic        = None
        fit.srmr       = pls_result.srmr
        if pls_result.r_squared:
            fit.r_squared = {k: round(v, 4) for k, v in pls_result.r_squared.items()}

        _emit(log_fn, "step", "Computing measurement validity (AVE · CR · α · Fornell-Larcker)")
        measurement = parsed.get("measurement", {})

        # Convert outer_loadings dict-of-dicts → {lv: [λ, ...]} for reuse
        loadings: dict = pls_loadings_to_list(pls_result.outer_loadings, measurement)

        try:
            if loadings:
                fit.ave = _compute_ave(loadings)
            else:
                warnings.append("No PLS loadings found; skipping AVE.")
        except Exception as e:
            warnings.append(f"Could not compute AVE: {e}")

    # ── CB-SEM / WLS branch ───────────────────────────────────────────────────
    else:
        # TC-31: semopy's SLSQP optimizer sporadically hits a degenerate
        # starting point on real datasets (observed: 3/4 runs fail in < 2 s
        # on HS1939).  Strategy: two SLSQP attempts with fresh Model objects
        # (resolves transient thread/state issues), then an L-BFGS-B fallback
        # for persistent ill-conditioning.  WLS uses a single attempt because
        # its objective is fundamentally different and solver-swapping is not
        # meaningful there.
        _last_cbsem_exc: Optional[Exception] = None
        sem_model = None

        if estimator == "WLS":
            try:
                sem_model = Model(syntax)
                sem_model.fit(df, obj="WLS")
            except Exception as e:
                raise ValueError(f"Model did not converge: {e}")
        else:
            # TC-52 guard: near-constant columns make S rank-deficient, causing
            # all solver attempts to fail.  Surface a clear error immediately.
            # Scope to model indicator columns only (not all df.columns) — extra
            # columns such as HS1939's 'school'/'grade' strings would raise
            # TypeError: Cannot perform reduction 'std' with string dtype (TC-31).
            _indicator_cols = [c for c in parsed.get("observed_vars", []) if c in df.columns]
            _near_const = [
                c for c in _indicator_cols
                if pd.api.types.is_numeric_dtype(df[c]) and df[c].std(ddof=1) < 1e-6
            ]
            if _near_const:
                raise ValueError(
                    f"CB-SEM cannot fit near-constant columns (std ≈ 0): "
                    f"{_near_const}. Check data scaling or switch to PLS-SEM."
                )
            _cb_solvers = ["SLSQP", "SLSQP", "L-BFGS-B", "SLSQP_standardised"]
            # _df_fit tracks which dataframe was actually used for fitting;
            # it may be z-scored (SLSQP_standardised) and must be forwarded
            # to all downstream calls (calc_stats, predict_factors, _compute_srmr)
            # so they operate on the same data the model internals reference.
            _df_fit = df
            for _att, _slv in enumerate(_cb_solvers):
                try:
                    _df_attempt = df
                    _solver  = _slv
                    if _slv == "SLSQP_standardised":
                        # TC-52 fix: column-wise z-scoring eliminates scale-driven
                        # ill-conditioning of the sample covariance matrix, which is
                        # the proximate trigger for scipy.linalg.eigvalsh non-convergence
                        # when correlated observed-indicator residuals (e.g. POLDEM y~~y
                        # constraints) push off-diagonal Theta eigenvalues near zero.
                        _std = df.std(ddof=1).replace(0, 1)
                        _df_attempt = (df - df.mean()) / _std
                        _solver = "SLSQP"
                    sem_model = Model(syntax)
                    try:
                        sem_model.fit(_df_attempt, solver=_solver)
                    except TypeError:
                        # Older semopy versions: fit() has no solver kwarg
                        sem_model.fit(_df_attempt)
                    _df_fit = _df_attempt   # record the dataframe used for this fit
                    _last_cbsem_exc = None
                    break                        # success — exit retry loop
                except Exception as _e:
                    _last_cbsem_exc = _e
                    try:
                        _emit(log_fn, "warn",
                              f"CB-SEM attempt {_att + 1}/{len(_cb_solvers)} "
                              f"failed (solver={_slv}): {str(_e)[:100]}")
                    except Exception: # nosec B110
                        pass   # never let a logging failure mask the real error
                    sem_model = None
            if sem_model is None:
                raise ValueError(
                    f"Model did not converge after {len(_cb_solvers)} "
                    f"attempts: {_last_cbsem_exc}"
                )

        algo_label = (
            "CB-SEM" if algorithm == "cb"
            else ("WLS" if algorithm == "wls" else "CB-SEM (ML)")
        )

        _emit(log_fn, "step", "Extracting parameter estimates")
        try:
            params_df = sem_model.inspect()
        except Exception as _insp_exc:
            raise ValueError(
                f"CB-SEM model converged but parameter extraction failed "
                f"(semopy inspect() error): {_insp_exc}. "
                "Try switching to PLS-SEM or check the model specification."
            ) from _insp_exc

        parameters = []
        for _, row in params_df.iterrows():
            est = _safe_float(row.get("Estimate", row.get("estimate", 0.0)), 0.0)
            lhs = str(row.get("lval", row.get("lhs", "")))
            op  = str(row.get("op",   "~"))
            rhs = str(row.get("rval", row.get("rhs", "")))

            if op == "~~":
                # Covariance / variance rows: semopy does not produce SE, z, or p
                # for these.  Storing 0/1 sentinels would be academically misleading
                # (PhD users might cite them).  Store None so the UI renders "—".
                parameters.append(PathParameter(
                    lhs=lhs, op=op, rhs=rhs,
                    estimate=est,
                    std_error=None,
                    z_value=None,
                    p_value=None,
                    ci_lower=None,
                    ci_upper=None,
                    significant=False,
                ))
            else:
                # =~ (loading) and ~ (structural) rows: real ML inference
                se    = _safe_float(row.get("Std. Err.", row.get("std_err", None)))
                z     = _safe_float(row.get("z-Value",  row.get("z_value",  None)))
                p     = _safe_float(row.get("p-Value",  row.get("p_value",  None)))
                ci_lo = round(est - 1.96 * se, 6) if se is not None else None
                ci_hi = round(est + 1.96 * se, 6) if se is not None else None
                parameters.append(PathParameter(
                    lhs=lhs, op=op, rhs=rhs,
                    estimate=est,
                    std_error=se,
                    z_value=z,
                    p_value=p,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    significant=_p_to_sig(p),
                ))

        _emit(log_fn, "step", "Computing fit indices (CFI · TLI · RMSEA · SRMR · AIC · BIC)")
        fit = FitIndices()

        # ── Standardised estimates (std.all) ──────────────────────────────────
        # Compute once here; populates =~ loadings and ~~ correlations.
        try:
            std_map = _compute_std_estimates(params_df, parsed.get("measurement", {}))
            for param in parameters:
                key = (param.lhs, param.op, param.rhs)
                if key in std_map:
                    param.std_estimate = round(std_map[key], 6)
        except Exception as _std_err:
            warnings.append(f"Could not compute standardised estimates: {_std_err}")

        # ── Standard fit statistics ───────────────────────────────────────────
        try:
            from semopy import calc_stats
            stats = calc_stats(sem_model)

            def gs(key):
                # Normalize: lowercase + collapse spaces/hyphens to underscore
                def norm(s):
                    return s.lower().replace("-", "_").replace(" ", "_")
                target = norm(key)
                for k, v in stats.items():
                    if norm(k) == target:
                        return _safe_float(v)
                return None

            # ── Debug: log all keys returned by calc_stats ───────────────────
            _emit(log_fn, "info", f"calc_stats keys: {list(stats.keys())}")

            def _first_match(*keys):
                """Return the first non-None match; handles 0.0 correctly (or-chain discards it)."""
                for k in keys:
                    v = gs(k)
                    if v is not None:
                        return v
                return None

            fit.cfi        = _first_match("CFI")
            fit.tli        = _first_match("TLI", "NNFI", "TLI_robust", "tli", "nnfi")
            fit.rmsea      = _first_match("RMSEA")
            fit.srmr       = _first_match("SRMR")
            if fit.srmr is None:
                fit.srmr   = _compute_srmr(sem_model, _df_fit, params_df, parsed)
            fit.chi_square = _first_match("chi2", "chi_square", "Chi2", "Chi-square")
            _df_raw        = _first_match("dof", "DoF", "df", "DF")
            fit.df         = int(_df_raw) if _df_raw is not None else None
            fit.p_value    = _first_match(
                "chi2_p_value", "chi2_p-value", "chi2 p-value",
                "p_value", "p-value", "pvalue", "p value",
            )
            _emit(log_fn, "info",
                  f"Raw fit: chi2={fit.chi_square} df={fit.df} p={fit.p_value} "
                  f"cfi={fit.cfi} tli={fit.tli} rmsea={fit.rmsea}")

            # ── p-value fallback: compute from chi2 and df via scipy ─────────
            # NOTE: _safe_float rounds to 6 dp, so 1.8e-9 becomes 0.0 (falsy).
            # Use float() directly here to preserve full precision for Pydantic.
            if fit.p_value is None and fit.chi_square is not None and fit.df:
                try:
                    from scipy.stats import chi2 as _scipy_chi2
                    raw_p = float(_scipy_chi2.sf(float(fit.chi_square), int(fit.df)))
                    if not (np.isnan(raw_p) or np.isinf(raw_p)):
                        fit.p_value = raw_p
                    _emit(log_fn, "info", f"p-value via scipy fallback: {fit.p_value}")
                except Exception as _pe:
                    _emit(log_fn, "warn", f"scipy p-value fallback failed: {_pe}")
            fit.aic = _first_match("AIC", "aic")
            fit.bic = _first_match("BIC", "bic")

            r2: dict = {}
            try:
                factor_scores = sem_model.predict_factors(_df_fit)
                preds_by_lhs: dict = {}
                for rel in parsed["structural"]:
                    preds_by_lhs.setdefault(rel["lhs"], []).append(rel["rhs"])
                for lhs, rhs_list in preds_by_lhs.items():
                    if lhs not in factor_scores.columns:
                        continue
                    predictors = [r for r in rhs_list if r in factor_scores.columns]
                    if not predictors:
                        continue
                    y = factor_scores[lhs].values
                    X = np.column_stack(
                        [np.ones(len(y))] + [factor_scores[r].values for r in predictors]
                    )
                    try:
                        beta = np.linalg.lstsq(X, y, rcond=None)[0]
                        y_hat = X @ beta
                        ss_res = float(((y - y_hat) ** 2).sum())
                        ss_tot = float(((y - y.mean()) ** 2).sum())
                        r2[lhs] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
                    except Exception as _e:  # B110
                        logger.debug("Non-critical exception suppressed: %s", _e)
                        pass
            except Exception as _e:  # B110
                logger.debug("Non-critical exception suppressed: %s", _e)
                pass
            if r2:
                fit.r_squared = r2

        except Exception as e:
            warnings.append(f"Could not compute fit statistics: {e}")

        _emit(log_fn, "step", "Computing measurement validity (AVE · CR · α · Fornell-Larcker)")
        # ── Measurement validity metrics ──────────────────────────────────────
        measurement = parsed.get("measurement", {})

        loadings: dict = {}
        try:
            loadings = _extract_loadings(params_df, measurement, _df_fit)
            if loadings:
                fit.ave = _compute_ave(loadings)
            else:
                warnings.append("No loadings found for measurement LVs; skipping AVE.")
        except Exception as e:
            warnings.append(f"Could not compute AVE: {e}")

    try:
        if loadings:
            fit.composite_reliability = _compute_composite_reliability(loadings)
    except Exception as e:
        warnings.append(f"Could not compute composite reliability: {e}")

    try:
        if measurement:
            fit.cronbach_alpha = _compute_cronbach_alpha(df, measurement)
    except Exception as e:
        warnings.append(f"Could not compute Cronbach α: {e}")

    try:
        if fit.ave and measurement:
            fl_matrix, fl_pass = _compute_fornell_larcker(fit.ave, df, measurement)
            fit.fornell_larcker = fl_matrix
            fit.fornell_larcker_pass = fl_pass
        elif not fit.ave:
            warnings.append("Skipping Fornell-Larcker: AVE could not be computed.")
    except Exception as e:
        warnings.append(f"Could not compute Fornell-Larcker matrix: {e}")

    fit = _fit_verdict(fit)

    _emit(log_fn, "step", "Computing VIF multicollinearity and Cohen's f² effect sizes")
    # ── v0.4: VIF and f² ─────────────────────────────────────────────────────
    vif_entries: list[VIFEntry] = []
    try:
        vif_entries = compute_vif(df, model_syntax)
    except Exception as e:
        warnings.append(f"Could not compute VIF: {e}")

    f2_entries: list[F2Entry] = []
    try:
        f2_entries = compute_f2(df, model_syntax)
    except Exception as e:
        warnings.append(f"Could not compute f²: {e}")

    if bootstrap_n > 0:
        _emit(log_fn, "step", f"Computing outer weight significance via {bootstrap_n} bootstrap samples")
    outer_weight_entries: list[OuterWeightEntry] = []
    try:
        if use_pls:
            # For PLS-SEM: point-only from the already-computed pls_result;
            # bootstrap outer weights handled in the bootstrap back-fill below.
            outer_weight_entries = _pls_outer_weight_entries_from_result(pls_result)
        elif bootstrap_n > 0:
            outer_weight_entries = compute_outer_weight_significance(
                df, model_syntax, n=bootstrap_n
            )
        else:
            outer_weight_entries = _compute_outer_weights_point_only(
                df, model_syntax
            )
    except Exception as e:
        warnings.append(f"Could not compute outer weight significance: {e}")

    # ── Significance back-fill from bootstrap CIs ────────────────────────────
    # Triggers when bootstrap was run AND any structural path has no real
    # p-value (p == 1.0 sentinel). This covers: PLS-SEM (no analytical p),
    # PLS falling back to CB-SEM (use_pls stays False), and any estimator
    # where semopy returns NaN/None p-values.
    structural_vars_set = {r["lhs"] for r in parsed.get("structural", [])} | \
                          {r["rhs"] for r in parsed.get("structural", [])}
    structural_params = [p for p in parameters
                         if p.op == "~"
                         and p.lhs in structural_vars_set
                         and p.rhs in structural_vars_set]
    missing_pvals = any(p.p_value is None or p.p_value >= 0.999 for p in structural_params)

    if bootstrap_n > 0 and (missing_pvals or use_pls):  # PLS always needs bootstrap SEs
        try:
            bs_result_tmp = run_bootstrap(df, model_syntax, n=bootstrap_n,
                                          algorithm=algorithm)
            bs_sig_map: dict[tuple[str, str, str], tuple[bool, float, float]] = {}
            for bp in bs_result_tmp.parameters:
                key = (str(bp.get("lhs", "")), str(bp.get("op", "")),
                       str(bp.get("rhs", "")))
                bs_sig_map[key] = (
                    bool(bp.get("significant", False)),
                    float(bp.get("ci_lower_95", 0)),
                    float(bp.get("ci_upper_95", 0)),
                )
            for param in parameters:
                key = (param.lhs, param.op, param.rhs)
                if key in bs_sig_map:
                    sig, ci_lo, ci_hi = bs_sig_map[key]
                    param.significant = sig
                    param.p_value = 0.001 if sig else 0.999
                    if param.ci_lower is None:
                        param.ci_lower = round(ci_lo, 6)
                    if param.ci_upper is None:
                        param.ci_upper = round(ci_hi, 6)
        except Exception as e:
            warnings.append(f"Could not back-fill significance from bootstrap: {e}")

    _emit(log_fn, "ok", f"Model fitted · {len(parameters)} parameters · algorithm: {algo_label}")

    summary = _build_summary(
        algo_label           = algo_label,
        n_obs                = len(df),
        bootstrap_n          = bootstrap_n,
        parameters           = parameters,
        fit                  = fit,
        parsed               = parsed,
        f2_entries           = f2_entries,
        outer_weight_entries = outer_weight_entries,
    )

    return ModelResult(
        algorithm=algo_label,
        n_obs=len(df),
        n_params=len(parameters),
        converged=True,
        parameters=parameters,
        fit=fit,
        latent_variables=parsed["latent_vars"],
        observed_variables=parsed["observed_vars"],
        vif=vif_entries or None,
        f2=f2_entries or None,
        outer_weights=outer_weight_entries or None,
        warnings=warnings,
        summary=summary,
    )



# ── Results summary builder ─────────────────────────────────────────────────────────────────────

def _build_summary(
    algo_label:           str,
    n_obs:                int,
    bootstrap_n:          int,
    parameters:           list,
    fit:                  FitIndices,
    parsed:               dict,
    f2_entries:           list,
    outer_weight_entries: list,
) -> ModelSummary:
    """
    Build a ModelSummary from already-computed engine outputs.
    Called at the very end of fit_model() so it never blocks the hot path.
    All errors are swallowed — a missing/incomplete summary is non-fatal.
    """
    measurement = parsed.get("measurement", {})

    # ── f2 lookup {(lhs, rhs): F2Entry} ──────────────────────────────────────
    f2_map: dict = {}
    for entry in (f2_entries or []):
        f2_map[(entry.lhs, entry.rhs)] = entry

    # ── outer loading lookup {lv: [loading, ...]} ─────────────────────────────
    # Prefer outer_weight_entries (reflective outer loadings stored there).
    # Fall back to fit.ave derivation if unavailable.
    loading_map: dict[str, list[float]] = {}
    for ow in (outer_weight_entries or []):
        loading_map.setdefault(ow.lv, []).append(ow.estimate)

    # ── 1. Structural path summaries ──────────────────────────────────────────
    structural_vars = {r["lhs"] for r in parsed.get("structural", [])} |                       {r["rhs"] for r in parsed.get("structural", [])}

    struct_params = {
        (p.lhs, p.rhs): p
        for p in parameters
        if p.op == "~" and p.lhs in structural_vars and p.rhs in structural_vars
    }

    structural_paths: list = []
    for rel in parsed.get("structural", []):
        lhs, rhs = rel["lhs"], rel["rhs"]
        p = struct_params.get((lhs, rhs))
        if p is None:
            continue

        # t-stat: prefer p.z_value (CB-SEM) else derive from bootstrap SE
        t_stat = _safe_float(p.z_value) if p.z_value is not None and p.z_value != 0.0 else None
        if t_stat is None and p.std_error and p.std_error > 0:
            t_stat = round(p.estimate / p.std_error, 4)

        f2e = f2_map.get((lhs, rhs))
        structural_paths.append(StructuralPathSummary(
            from_var    = rhs,
            to_var      = lhs,
            beta        = round(p.estimate, 4),
            t_stat      = round(t_stat, 4) if t_stat else None,
            p_value     = round(p.p_value, 4) if p.p_value is not None and p.p_value <= 0.999 else None,
            ci_lower_95 = round(p.ci_lower, 4) if p.ci_lower is not None else None,
            ci_upper_95 = round(p.ci_upper, 4) if p.ci_upper is not None else None,
            significant = p.significant,
            f2          = round(f2e.f2, 4) if f2e else None,
            f2_label    = f2e.effect if f2e else None,
        ))

    # ── 2. Construct validity summaries ───────────────────────────────────────
    ave_map   = fit.ave or {}
    cr_map    = fit.composite_reliability or {}
    alpha_map = fit.cronbach_alpha or {}

    construct_validity: list = []
    for lv, indicators in measurement.items():
        loadings = loading_map.get(lv, [])
        avg_lam  = round(float(np.mean(loadings)), 4)  if loadings else None
        min_lam  = round(float(np.min(loadings)),  4)  if loadings else None
        ave_val  = round(ave_map.get(lv),   4)         if lv in ave_map  else None
        cr_val   = round(cr_map.get(lv),    4)         if lv in cr_map   else None
        alp_val  = round(alpha_map.get(lv), 4)         if lv in alpha_map else None
        ave_sqrt = round(float(ave_val) ** 0.5, 4)     if ave_val is not None else None

        construct_validity.append(ConstructValiditySummary(
            construct_name        = lv,
            n_indicators          = len([i for i in indicators if i in parsed.get("observed_vars", [])]),
            avg_loading           = avg_lam,
            min_loading           = min_lam,
            ave                   = ave_val,
            ave_sqrt              = ave_sqrt,
            composite_reliability = cr_val,
            cronbach_alpha        = alp_val,
            ave_ok                = (ave_val  >= 0.50) if ave_val  is not None else None,
            cr_ok                 = (cr_val   >= 0.70) if cr_val   is not None else None,
            alpha_ok              = (alp_val  >= 0.70) if alp_val  is not None else None,
        ))

    all_loadings_ok = (
        all(c.avg_loading >= 0.70 for c in construct_validity if c.avg_loading is not None)
        if construct_validity else None
    )

    # ── 3. Fit ────────────────────────────────────────────────────────────────
    srmr = fit.srmr
    srmr_ok = (srmr <= 0.08) if srmr is not None else None

    # ── 4. Verdict ────────────────────────────────────────────────────────────
    issues: list[str] = []
    passes: list[str] = []

    # Structural significance
    # Guard: if bootstrap wasn't run and all p-values are the 1.0 sentinel
    # (semopy version quirk), skip significance verdict rather than false-flagging.
    pvals_available = any(
        p.p_value is not None and p.p_value < 0.999
        for p in structural_paths
    )
    ci_available = any(
        p.ci_lower_95 is not None and p.ci_upper_95 is not None
        for p in structural_paths
    )
    n_sig = sum(1 for p in structural_paths if p.significant)
    n_tot = len(structural_paths)
    if n_tot and (pvals_available or ci_available):
        if n_sig == n_tot:
            passes.append(f"all {n_tot} path(s) significant")
        else:
            issues.append(f"{n_tot - n_sig}/{n_tot} path(s) non-significant")
    elif n_tot and bootstrap_n == 0:
        passes.append(f"{n_tot} structural path(s) estimated — run with bootstrap for significance tests")

    # Measurement quality
    failed_ave = [c.construct_name for c in construct_validity if c.ave_ok is False]
    failed_cr  = [c.construct_name for c in construct_validity if c.cr_ok  is False]
    if not failed_ave and construct_validity:
        passes.append("AVE ≥ 0.50 for all constructs")
    elif failed_ave:
        issues.append(f"AVE < 0.50: {', '.join(failed_ave)}")
    if not failed_cr and construct_validity:
        passes.append("CR ≥ 0.70")
    elif failed_cr:
        issues.append(f"CR < 0.70: {', '.join(failed_cr)}")

    # Discriminant validity
    if fit.fornell_larcker_pass is True:
        passes.append("Fornell-Larcker criterion met")
    elif fit.fornell_larcker_pass is False:
        issues.append("Fornell-Larcker criterion failed")

    # Fit
    if srmr is not None:
        if srmr <= 0.08:
            passes.append(f"SRMR = {srmr:.3f} (acceptable)")
        else:
            issues.append(f"SRMR = {srmr:.3f} (> 0.08)")

    if issues:
        verdict = "Concerns: " + "; ".join(issues) + ". " + (", ".join(passes) + "." if passes else "")
    elif passes:
        verdict = "Good fit: " + "; ".join(passes) + "."
    else:
        verdict = "Results computed — review tables for interpretation."

    return ModelSummary(
        algorithm            = algo_label,
        n_obs                = n_obs,
        bootstrap_n          = bootstrap_n,
        structural_paths     = structural_paths,
        construct_validity   = construct_validity,
        fornell_larcker_pass = fit.fornell_larcker_pass,
        all_loadings_ok      = all_loadings_ok,
        srmr                 = srmr,
        srmr_ok              = srmr_ok,
        r_squared            = fit.r_squared,
        cfi                  = fit.cfi,
        rmsea                = fit.rmsea,
        verdict              = verdict,
    )


# ── Bootstrapping ─────────────────────────────────────────────────────────────

def run_bootstrap(
    df: pd.DataFrame,
    model_syntax: str,
    n: int = 500,
    algorithm: str = "pls",
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> BootstrapResult:
    parsed = parse_lavaan(model_syntax)
    syntax = build_semopy_syntax(parsed)
    rng = np.random.default_rng(seed)

    all_estimates = []
    converged = 0
    _emit(log_fn, "step", f"Bootstrap: running {n} resamples (seed={seed})")
    _t0 = time.time()

    # ── Decide which estimator to use per resample ────────────────────────────
    use_pls_bs = (algorithm == "pls")

    if use_pls_bs:
        # PLS bootstrap: row-resample, run PLSEstimator, collect path coefs + loadings
        from app.pls import PLSEstimator

        # We need stable parameter ordering so CIs line up.
        # Order: outer loadings (=~) alphabetically, then structural (~) alphabetically.
        def _pls_param_order(pls_res) -> list[tuple[str, str, str]]:
            order = []
            for lv in sorted(pls_res.outer_loadings):
                for ind in sorted(pls_res.outer_loadings[lv]):
                    order.append((lv, "=~", ind))
            for lhs in sorted(pls_res.path_coefficients):
                for rhs in sorted(pls_res.path_coefficients[lhs]):
                    order.append((lhs, "~", rhs))
            return order

        # Full-data run to fix the parameter order
        try:
            pls_full = PLSEstimator().fit(df, parsed)
        except Exception as e:
            raise ValueError(f"PLS full-data fit failed before bootstrap: {e}")
        param_order = _pls_param_order(pls_full)

        def _pls_vector(pls_res, order):
            vals = []
            for lhs, op, rhs in order:
                if op == "=~":
                    vals.append(pls_res.outer_loadings.get(lhs, {}).get(rhs, np.nan))
                else:
                    vals.append(pls_res.path_coefficients.get(lhs, {}).get(rhs, np.nan))
            return np.array(vals, dtype=float)

        full_vec = _pls_vector(pls_full, param_order)

        for _bi in range(n):
            if _bi > 0 and _bi % 100 == 0:
                _emit(log_fn, "info", f"  Bootstrap: {_bi}/{n} samples · {converged} converged so far")
            sample = df.sample(frac=1, replace=True, random_state=int(rng.integers(1e6)))
            try:
                pls_bs = PLSEstimator().fit(sample, parsed)
                all_estimates.append(_pls_vector(pls_bs, param_order))
                converged += 1
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

        labels = [{"lhs": lhs, "op": op, "rhs": rhs} for lhs, op, rhs in param_order]
        orig_vec = full_vec

    else:
        # CB-SEM / WLS bootstrap: semopy Model per resample
        try:
            from semopy import Model
        except ImportError:
            raise RuntimeError("semopy is not installed.")

        for _bi in range(n):
            if _bi > 0 and _bi % 100 == 0:
                _emit(log_fn, "info", f"  Bootstrap: {_bi}/{n} samples · {converged} converged so far")
            sample = df.sample(frac=1, replace=True, random_state=int(rng.integers(1e6)))
            try:
                m = Model(syntax)
                m.fit(sample)
                p = m.inspect()
                row_vals = p["Estimate"].values if "Estimate" in p.columns else p["estimate"].values
                all_estimates.append(row_vals)
                converged += 1
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

        try:
            orig = Model(syntax)
            orig.fit(df)
            orig_p = orig.inspect()
            labels = [
                {"lhs": str(r.get("lval", r.get("lhs", ""))),
                 "op": str(r.get("op", "~")),
                 "rhs": str(r.get("rval", r.get("rhs", "")))}
                for _, r in orig_p.iterrows()
            ]
            est_col = "Estimate" if "Estimate" in orig_p.columns else "estimate"
            orig_vec = orig_p[est_col].values
        except Exception as _e:
            labels   = [{"lhs": f"param_{i}", "op": "~", "rhs": ""} for i in range(len(all_estimates[0]) if all_estimates else 0)]
            orig_vec = np.zeros(len(labels))

    _elapsed = round(time.time() - _t0, 1)
    _emit(log_fn, "ok", f"Bootstrap complete · {converged}/{n} converged ({round(converged/n*100,1)}%) · {_elapsed}s")
    if not all_estimates:
        raise ValueError("No bootstrap samples converged.")

    est_array = np.array(all_estimates)
    bs_se     = np.std(est_array,        axis=0, ddof=1)
    ci_lo     = np.percentile(est_array, 2.5,   axis=0)
    ci_hi     = np.percentile(est_array, 97.5,  axis=0)
    bs_mean   = np.mean(est_array,               axis=0)

    parameters = []
    for i, lab in enumerate(labels):
        if i >= len(bs_se):
            break
        pe = float(orig_vec[i]) if i < len(orig_vec) else 0.0
        parameters.append({
            **lab,
            "estimate":     round(pe, 6),
            "bs_mean":      round(float(bs_mean[i]), 6),
            "bs_se":        round(float(bs_se[i]),   6),
            "ci_lower_95":  round(float(ci_lo[i]),   6),
            "ci_upper_95":  round(float(ci_hi[i]),   6),
            "significant":  not (ci_lo[i] <= 0 <= ci_hi[i]),
        })

    return BootstrapResult(
        n_samples=n,
        parameters=parameters,
        converged_pct=round(converged / n * 100, 1),
    )


# ── HTMT ──────────────────────────────────────────────────────────────────────

def compute_htmt(df: pd.DataFrame, model_syntax: str) -> HTMTResult:
    parsed = parse_lavaan(model_syntax)
    measurement = parsed["measurement"]
    lvs = list(measurement.keys())

    if len(lvs) < 2:
        raise ValueError("HTMT requires at least 2 latent variables.")

    corr = df.corr(numeric_only=True)

    def mean_abs_corr(vars_a, vars_b, same=False):
        vals = []
        for a in vars_a:
            for b in vars_b:
                if same and a == b:
                    continue
                if a in corr.columns and b in corr.columns:
                    vals.append(abs(corr.loc[a, b]))
        return np.mean(vals) if vals else np.nan

    entries = []
    for i, lv_a in enumerate(lvs):
        for lv_b in lvs[i + 1:]:
            inds_a = measurement[lv_a]
            inds_b = measurement[lv_b]
            # Single-indicator constructs have no within-LV variance to compare;
            # HTMT is undefined for them — skip rather than emitting a 9999 sentinel
            # that inflates max_htmt and triggers false advisory failures.
            if len(inds_a) < 2 or len(inds_b) < 2:
                continue
            cross = mean_abs_corr(inds_a, inds_b)
            within_a = mean_abs_corr(inds_a, inds_a, same=True)
            within_b = mean_abs_corr(inds_b, inds_b, same=True)
            denom = np.sqrt(within_a * within_b)
            htmt_val = cross / denom if denom > 0 else np.nan
            if np.isnan(htmt_val):
                continue
            entries.append(HTMTEntry(
                construct_a=lv_a,
                construct_b=lv_b,
                htmt=round(float(htmt_val), 4),
                acceptable=htmt_val < 0.90,
            ))

    return HTMTResult(
        matrix=entries,
        all_acceptable=all(e.acceptable for e in entries),
    )


# ── PLS outer weight helpers ──────────────────────────────────────────────────

def _pls_outer_weight_entries_from_result(pls_result) -> list[OuterWeightEntry]:
    """
    Build OuterWeightEntry list from a PLSResult (point estimates only).
    bs_mean mirrors the estimate; CIs are None until bootstrap back-fill runs.
    """
    from app.pls import PLSResult  # local import to avoid circular risk
    entries: list[OuterWeightEntry] = []
    for lv, ind_map in pls_result.outer_loadings.items():
        for ind, loading in ind_map.items():
            # Also grab the outer weight (for formative; loadings for reflective)
            weight = pls_result.outer_weights.get(lv, {}).get(ind, loading)
            entries.append(OuterWeightEntry(
                lv=lv,
                indicator=ind,
                estimate=round(loading, 6),   # report loading (reflective convention)
                bs_mean=round(loading, 6),
                bs_se=0.0,
                ci_lower_95=0.0,
                ci_upper_95=0.0,
                t_stat=None,
                significant=False,            # back-filled by bootstrap if requested
            ))
    return entries


# ── Outer weight significance (CB-SEM) ───────────────────────────────────────

def _compute_outer_weights_point_only(
    df: pd.DataFrame,
    model_syntax: str,
) -> list[OuterWeightEntry]:
    """
    Outer weight / loading point estimates without bootstrapping.

    Used when bootstrap_n == 0 so the canvas still receives labels for every
    measurement edge even if no significance information is available.
    bs_mean mirrors the point estimate; bs_se / CIs are zeroed; significant is
    left False (unknown without a bootstrap distribution).
    """
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed.")

    parsed      = parse_lavaan(model_syntax)
    syntax      = build_semopy_syntax(parsed)
    measurement = parsed.get("measurement", {})
    if not measurement:
        return []

    m = Model(syntax)
    try:
        m.fit(df)
    except Exception as _fit_exc:
        raise ValueError(
            f"_compute_outer_weights_point_only: model fit failed: {_fit_exc}"
        ) from _fit_exc
    try:
        loadings = _extract_loadings(m.inspect(), measurement, df)
    except Exception as _insp_exc:
        raise ValueError(
            f"_compute_outer_weights_point_only: inspect() failed: {_insp_exc}"
        ) from _insp_exc

    entries: list[OuterWeightEntry] = []
    for lv, indicators in measurement.items():
        lv_lams = loadings.get(lv, [])
        inds    = [i for i in indicators if i in df.columns]
        for k, ind in enumerate(inds):
            if k >= len(lv_lams):
                continue
            pe = lv_lams[k]
            entries.append(OuterWeightEntry(
                lv=lv,
                indicator=ind,
                estimate=round(pe, 6),
                bs_mean=round(pe, 6),
                bs_se=0.0,
                ci_lower_95=0.0,
                ci_upper_95=0.0,
                t_stat=None,
                significant=False,
            ))
    return entries


def compute_outer_weight_significance(
    df: pd.DataFrame,
    model_syntax: str,
    n: int = 500,
    seed: int = 42,
) -> list[OuterWeightEntry]:
    """
    Bootstrap significance test for outer weights / loadings.

    For each indicator-LV pair in the measurement model:
      - Point estimate: loading/weight from the full-data fit
      - Bootstrap distribution: n re-fits on resampled data
      - Reports BS mean, SE, 95% percentile CI, t-stat = estimate / BS_SE
      - Significant when the 95% CI excludes zero

    Works for both reflective (outer loadings) and formative (outer weights)
    indicators. Uses the same variable-name lookup as _extract_loadings so it
    is robust to semopy's op-column inconsistencies across versions.
    """
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed.")

    parsed     = parse_lavaan(model_syntax)
    syntax     = build_semopy_syntax(parsed)
    measurement = parsed.get("measurement", {})

    if not measurement:
        return []

    # ── Point estimates from full-data fit ────────────────────────────────────
    m_full = Model(syntax)
    m_full.fit(df)
    full_loadings = _extract_loadings(m_full.inspect(), measurement, df)

    # Build ordered list of (lv, indicator) pairs we have estimates for
    pairs: list[tuple[str, str]] = []
    point_ests: list[float] = []
    for lv, indicators in measurement.items():
        lv_lams = full_loadings.get(lv, [])
        inds_with_data = [i for i in indicators if i in df.columns]
        for k, ind in enumerate(inds_with_data):
            if k < len(lv_lams):
                pairs.append((lv, ind))
                point_ests.append(lv_lams[k])

    if not pairs:
        return []

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    bs_collections: list[list[float]] = [[] for _ in pairs]
    rng = np.random.default_rng(seed)

    for _ in range(n):
        sample = df.sample(frac=1, replace=True,
                           random_state=int(rng.integers(1_000_000)))
        try:
            m_bs = Model(syntax)
            m_bs.fit(sample)
            bs_lams = _extract_loadings(m_bs.inspect(), measurement, sample)
            for idx, (lv, ind) in enumerate(pairs):
                lv_lams_bs = bs_lams.get(lv, [])
                inds_bs = [i for i in measurement[lv] if i in sample.columns]
                k = inds_bs.index(ind) if ind in inds_bs else -1
                if 0 <= k < len(lv_lams_bs):
                    bs_collections[idx].append(lv_lams_bs[k])
        except Exception as _e:  # B112
            logger.debug("Non-critical exception suppressed: %s", _e)
            continue

    # ── Assemble results ──────────────────────────────────────────────────────
    entries: list[OuterWeightEntry] = []
    for idx, (lv, ind) in enumerate(pairs):
        pe  = point_ests[idx]
        bs  = bs_collections[idx]
        if len(bs) < 2:
            continue
        bs_mean = float(np.mean(bs))
        bs_se   = float(np.std(bs, ddof=1))
        ci_lo   = float(np.percentile(bs, 2.5))
        ci_hi   = float(np.percentile(bs, 97.5))
        t_stat  = _safe_float(pe / bs_se) if bs_se > 0 else None
        entries.append(OuterWeightEntry(
            lv=lv,
            indicator=ind,
            estimate=round(pe, 6),
            bs_mean=round(bs_mean, 6),
            bs_se=round(bs_se, 6),
            ci_lower_95=round(ci_lo, 6),
            ci_upper_95=round(ci_hi, 6),
            t_stat=t_stat,
            significant=not (ci_lo <= 0 <= ci_hi),
        ))
    return entries


# ── VIF ───────────────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, model_syntax: str) -> list[VIFEntry]:
    """
    Variance Inflation Factor for each indicator within each LV block.
    For indicator i: VIF_i = 1 / (1 − R²_i)
    where R²_i = R² from regressing x_i on all other indicators in the same block.

    Useful for diagnosing multicollinearity in formative measurement models.
    Threshold: VIF < 5.0 is acceptable; < 3.3 is the strict PLS-SEM standard.
    """
    parsed = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    entries: list[VIFEntry] = []

    for lv, indicators in measurement.items():
        cols = [c for c in indicators if c in df.columns]
        if len(cols) < 2:
            if cols:
                entries.append(VIFEntry(lv=lv, indicator=cols[0], vif=1.0, acceptable=True))
            continue
        X = df[cols].dropna().values.astype(float)
        for i, ind in enumerate(cols):
            try:
                y = X[:, i]
                others = np.delete(X, i, axis=1)
                X_int = np.column_stack([np.ones(len(y)), others])
                beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
                y_pred = X_int @ beta
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = min(max(1.0 - ss_res / ss_tot, 0.0), 0.9999) if ss_tot > 0 else 0.0
                vif = _safe_float(1.0 / (1.0 - r2))
            except Exception as _e:
                vif = None
            if vif is not None:
                entries.append(VIFEntry(lv=lv, indicator=ind, vif=vif, acceptable=vif < 5.0))

    return entries


# ── f² effect size ─────────────────────────────────────────────────────────────

def _r2_for_lv(sem_model, df: pd.DataFrame, lv: str) -> Optional[float]:
    """Extract R² for a given endogenous LV from a fitted semopy model."""
    try:
        pred = sem_model.predict(df)
        if lv in pred.columns and lv in df.columns:
            ss_res = float(((df[lv] - pred[lv]) ** 2).sum())
            ss_tot = float(((df[lv] - df[lv].mean()) ** 2).sum())
            if ss_tot > 0:
                return 1.0 - ss_res / ss_tot
    except Exception as _e:  # B110
        logger.debug("Non-critical exception suppressed: %s", _e)
        pass
    return None


def compute_f2(
    df: pd.DataFrame,
    model_syntax: str,
) -> list[F2Entry]:
    """
    Cohen's f² effect size for each structural path.
    f² = (R²_full − R²_reduced) / (1 − R²_full)

    Computes R² directly from OLS residuals (composite-score approach):
    each LV is represented by the unweighted mean of its indicators.
    This avoids dependence on semopy's predict() which is unreliable for CB-SEM.

    Benchmarks (Cohen 1988): negligible < 0.02, small ≥ 0.02, medium ≥ 0.15, large ≥ 0.35.
    """
    parsed = parse_lavaan(model_syntax)
    structural = parsed.get("structural", [])
    measurement = parsed.get("measurement", {})
    if not structural:
        return []

    # Build LV composite scores: mean of indicators for each LV
    composites: dict[str, pd.Series] = {}
    for lv, indicators in measurement.items():
        cols = [c for c in indicators if c in df.columns]
        if cols:
            composites[lv] = df[cols].mean(axis=1)
        elif lv in df.columns:
            composites[lv] = df[lv]

    # For observed-only variables (pure path model)
    for rel in structural:
        for var in (rel["lhs"], rel["rhs"]):
            if var not in composites and var in df.columns:
                composites[var] = df[var]

    def ols_r2(y_series: pd.Series, x_series_list: list[pd.Series]) -> float:
        """R² from OLS regression of y on x_series_list."""
        if not x_series_list:
            return 0.0
        data = pd.concat([y_series] + x_series_list, axis=1).dropna()
        if len(data) < 2:
            return 0.0
        y = data.iloc[:, 0].values
        X = np.column_stack([np.ones(len(y))] + [data.iloc[:, i+1].values
                                                  for i in range(len(x_series_list))])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            return max(1.0 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0
        except Exception as _e:
            return 0.0

    # Group predictors by lhs
    from collections import defaultdict
    preds_by_lhs: dict[str, list[str]] = defaultdict(list)
    for rel in structural:
        preds_by_lhs[rel["lhs"]].append(rel["rhs"])

    entries: list[F2Entry] = []
    for rel in structural:
        lhs, rhs = rel["lhs"], rel["rhs"]
        if lhs not in composites or rhs not in composites:
            continue
        try:
            all_preds = preds_by_lhs[lhs]
            full_xs   = [composites[p] for p in all_preds if p in composites]
            reduced_xs = [composites[p] for p in all_preds if p in composites and p != rhs]

            r2_f = ols_r2(composites[lhs], full_xs)
            r2_r = ols_r2(composites[lhs], reduced_xs)

            denom   = 1.0 - r2_f
            f2_val  = max((r2_f - r2_r) / denom, 0.0) if denom > 0 else 0.0

            effect = ("large" if f2_val >= 0.35
                      else "medium" if f2_val >= 0.15
                      else "small"  if f2_val >= 0.02
                      else "negligible")

            entries.append(F2Entry(
                lhs=lhs, rhs=rhs,
                r2_full=round(r2_f, 6),
                r2_reduced=round(r2_r, 6),
                f2=round(f2_val, 6),
                effect=effect,
            ))
        except Exception as _e:  # B112
            logger.debug("Non-critical exception suppressed: %s", _e)
            continue

    return entries


# ── Indirect effects ───────────────────────────────────────────────────────────

def _build_coef_map(params_df: pd.DataFrame, structural_vars: set[str] | None = None) -> dict[tuple[str, str], float]:
    """
    Return {(rhs, lhs): coefficient} for structural (~) rows only.
    When structural_vars is provided, only rows where BOTH lhs and rhs are
    in that set are included — this filters out measurement loadings that
    semopy also writes with op='~'.
    """
    est_col = "Estimate" if "Estimate" in params_df.columns else "estimate"
    coef: dict[tuple[str, str], float] = {}
    for _, row in params_df.iterrows():
        op  = str(row.get("op", "~"))
        lhs = str(row.get("lval", row.get("lhs", "")))
        rhs = str(row.get("rval", row.get("rhs", "")))
        est = _safe_float(row.get(est_col))
        if est is not None and op == "~":
            if structural_vars is None or (lhs in structural_vars and rhs in structural_vars):
                coef[(rhs, lhs)] = est
    return coef


def _find_all_paths(
    graph: dict[str, list[str]],
    start: str,
    end: str,
    max_depth: int = 6,
) -> list[list[str]]:
    """All simple directed paths from start to end (no cycles)."""
    paths: list[list[str]] = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_depth + 1:
            continue
        for nxt in graph.get(node, []):
            if nxt in path:
                continue
            new_path = path + [nxt]
            if nxt == end:
                paths.append(new_path)
            else:
                stack.append((nxt, new_path))
    return paths


def compute_indirect_effects(
    df: pd.DataFrame,
    model_syntax: str,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> IndirectResult:
    """
    Decompose indirect effects for all variable pairs connected via paths ≥ 2 edges.
    Point estimate = product of path coefficients along each indirect path.
    Bootstrapped 95% percentile CIs computed when n_bootstrap > 0.
    Total effect = direct effect + sum of all indirect effects for each pair.
    """
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed.")

    parsed = parse_lavaan(model_syntax)
    syntax = build_semopy_syntax(parsed)
    structural = parsed.get("structural", [])

    if not structural:
        raise ValueError("No structural paths — indirect effects require a structural model.")

    # Build adjacency: rhs → [lhs, ...]
    graph: dict[str, list[str]] = {}
    for rel in structural:
        graph.setdefault(rel["rhs"], []).append(rel["lhs"])

    all_vars = list({v for rel in structural for v in (rel["lhs"], rel["rhs"])})
    structural_vars = set(all_vars)

    # Point estimates on full data
    m = Model(syntax)
    m.fit(df)
    coef = _build_coef_map(m.inspect(), structural_vars)

    def path_product(path: list[str], coef_map: dict) -> Optional[float]:
        prod = 1.0
        for i in range(len(path) - 1):
            c = coef_map.get((path[i], path[i + 1]))
            if c is None:
                return None
            prod *= c
        return prod

    # Enumerate all indirect paths (≥ 3 nodes = ≥ 1 mediator)
    indirect_spec: list[tuple[str, str, list[str]]] = []
    for src in all_vars:
        for dst in all_vars:
            if src == dst:
                continue
            for path in _find_all_paths(graph, src, dst):
                if len(path) >= 3:
                    indirect_spec.append((src, dst, path))

    if not indirect_spec:
        raise ValueError("No indirect paths found in this model.")

    point_estimates = [path_product(path, coef) for _, _, path in indirect_spec]

    # Bootstrap
    bs_samples: list[list[float]] = [[] for _ in indirect_spec]
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            sample = df.sample(frac=1, replace=True,
                               random_state=int(rng.integers(1_000_000)))
            try:
                m_bs = Model(syntax)
                m_bs.fit(sample)
                c_bs = _build_coef_map(m_bs.inspect(), structural_vars)
                for j, (_, _, path) in enumerate(indirect_spec):
                    v = path_product(path, c_bs)
                    if v is not None:
                        bs_samples[j].append(v)
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

    # Total effects: direct + indirect
    total: dict[str, dict[str, float]] = {}
    for (rhs, lhs), c in coef.items():
        total.setdefault(rhs, {})[lhs] = round(total.get(rhs, {}).get(lhs, 0.0) + c, 6)
    for j, (src, dst, _) in enumerate(indirect_spec):
        pe = point_estimates[j]
        if pe is not None:
            total.setdefault(src, {})[dst] = round(total.get(src, {}).get(dst, 0.0) + pe, 6)

    # Build output
    effects: list[IndirectEffect] = []
    for j, (src, dst, path) in enumerate(indirect_spec):
        pe = point_estimates[j]
        bs = bs_samples[j]
        bs_se  = _safe_float(np.std(bs, ddof=1)) if len(bs) > 1 else None
        ci_lo  = _safe_float(np.percentile(bs, 2.5))  if len(bs) > 1 else None
        ci_hi  = _safe_float(np.percentile(bs, 97.5)) if len(bs) > 1 else None
        sig    = (not (ci_lo <= 0 <= ci_hi)) if (ci_lo is not None and ci_hi is not None) else None
        effects.append(IndirectEffect(
            from_var=src,
            to_var=dst,
            through=path[1:-1],
            indirect_effect=_safe_float(pe) or 0.0,
            bs_se=bs_se,
            ci_lower_95=ci_lo,
            ci_upper_95=ci_hi,
            significant=sig,
        ))

    return IndirectResult(effects=effects, total_effects=total)


# ── CMB Marker Variable Analysis ──────────────────────────────────────────────

def compute_cmb(
    df: pd.DataFrame,
    model_syntax: str,
    marker_variable: str,
) -> CMBMarkerResult:
    """
    Common Method Bias (CMB) marker variable analysis (Lindell & Whitney 2001).

    A marker variable theoretically unrelated to the substantive constructs
    is correlated with every indicator. If the marker correlates highly with
    substantive indicators (r > 0.20), common method variance is a concern.

    Threshold: max |r| > 0.20 flags CMB concern.
    """
    parsed = parse_lavaan(model_syntax)
    observed = [v for v in parsed.get("observed_vars", []) if v in df.columns]

    if marker_variable not in df.columns:
        raise ValueError(f"Marker variable '{marker_variable}' not found in data.")
    if marker_variable in observed:
        raise ValueError(
            f"'{marker_variable}' is already a model indicator — "
            "choose a variable outside the structural model."
        )

    marker = df[marker_variable].dropna()
    correlations: dict[str, float] = {}
    for ind in observed:
        if ind in df.columns:
            r = _safe_float(df[ind].corr(marker))
            if r is not None:
                correlations[ind] = r

    if not correlations:
        raise ValueError("No valid correlations computed — check that indicators are numeric.")

    vals = list(correlations.values())
    mean_r = _safe_float(float(np.mean(np.abs(vals)))) or 0.0
    max_r  = _safe_float(float(np.max(np.abs(vals))))  or 0.0
    concern = max_r > 0.20

    note = (
        "CMB concern: marker variable correlates with substantive indicators "
        f"(max |r| = {max_r:.3f} > 0.20). Consider Harman single-factor or "
        "partial correlation controls."
        if concern else
        f"No CMB concern: max |r| = {max_r:.3f} ≤ 0.20 (Lindell & Whitney threshold)."
    )

    return CMBMarkerResult(
        marker_variable=marker_variable,
        correlations_with_substantive={k: round(v, 6) for k, v in correlations.items()},
        mean_marker_correlation=round(mean_r, 6),
        max_marker_correlation=round(max_r, 6),
        cmb_concern=concern,
        note=note,
    )


# ── Q² Blindfolding ────────────────────────────────────────────────────────────
# _build_composites lives in engine_utils.py


def compute_q2(
    df: pd.DataFrame,
    model_syntax: str,
    omission_distance: int = 7,
) -> list[Q2Entry]:
    """
    Stone-Geisser Q² via blindfolding (omission loop).

    For each endogenous LV with indicators:
      1. Set every D-th observation for that LV's indicators to NaN (omit).
      2. Predict using OLS on the composite of remaining predictors.
      3. Q² = 1 - SSE / SSO  where SSO = Σ(y - ȳ)² over omitted cells.

    Q² > 0 = predictive relevance.
    Benchmarks: small ≥ 0.02, medium ≥ 0.15, large ≥ 0.35.

    Uses composite-score OLS (not semopy) for speed and robustness.
    D = omission_distance (typically 5–10, must not be a multiple of n).
    """
    parsed     = parse_lavaan(model_syntax)
    structural = parsed.get("structural", [])
    measurement = parsed.get("measurement", {})

    if not structural:
        raise ValueError("Q² requires a structural model.")

    # Endogenous LVs: appear as lhs in structural paths
    endogenous = list({r["lhs"] for r in structural})

    # Predictor map: {lhs: [rhs, ...]}
    from collections import defaultdict
    preds_by_lhs: dict[str, list[str]] = defaultdict(list)
    for r in structural:
        preds_by_lhs[r["lhs"]].append(r["rhs"])

    composites = _build_composites(df, measurement, structural)
    entries: list[Q2Entry] = []

    for lv in endogenous:
        if lv not in composites:
            continue
        predictors = [p for p in preds_by_lhs[lv] if p in composites]
        if not predictors:
            continue

        y_full = composites[lv].values.astype(float)
        X_preds = np.column_stack([composites[p].values for p in predictors])
        n = len(y_full)

        # Ensure D does not divide n evenly (adjust if needed)
        D = omission_distance
        while n % D == 0 and D < n - 1:
            D += 1

        sse = 0.0
        sso = 0.0

        for start in range(D):
            omit_idx = np.arange(start, n, D)
            keep_idx = np.setdiff1d(np.arange(n), omit_idx)
            if len(keep_idx) < len(predictors) + 2:
                continue

            y_train = y_full[keep_idx]
            X_train = np.column_stack([np.ones(len(keep_idx)), X_preds[keep_idx]])
            y_test  = y_full[omit_idx]
            X_test  = np.column_stack([np.ones(len(omit_idx)),  X_preds[omit_idx]])

            try:
                beta   = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                y_pred = X_test @ beta
                sse   += float(np.sum((y_test - y_pred) ** 2))
                sso   += float(np.sum((y_test - np.mean(y_full)) ** 2))
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

        if sso <= 0:
            continue

        q2 = _safe_float(1.0 - sse / sso) or 0.0
        relevance = (
            "large"  if q2 >= 0.35 else
            "medium" if q2 >= 0.15 else
            "small"  if q2 >= 0.02 else
            "none"
        )
        entries.append(Q2Entry(
            lv=lv,
            q2=round(q2, 6),
            sse=round(sse, 4),
            sso=round(sso, 4),
            omission_distance=D,
            predictive_relevance=relevance,
        ))

    return entries


# ── PLSpredict + CVPAT ─────────────────────────────────────────────────────────

def compute_plspredict(
    df: pd.DataFrame,
    model_syntax: str,
    k_folds: int = 10,
    seed: int = 42,
) -> tuple[list[PLSPredictEntry], list[CVPATResult]]:
    """
    PLSpredict (Shmueli et al. 2019) + CVPAT (Liengaard et al. 2021).

    PLSpredict:
      k-fold cross-validation. Each fold: train on k-1 folds, predict
      held-out indicators of endogenous LVs. Compare RMSE/MAE against
      a simple LM baseline (predict using only means of exogenous composites).
      Q²_predict = 1 - (RMSE_model / RMSE_lm)²

    CVPAT:
      Computes the mean loss difference (LM loss - model loss) per observation.
      A one-sample t-test on this difference tests whether the model
      significantly outperforms the LM baseline.

    Both use composite-score OLS for speed and version-independence.
    """
    from collections import defaultdict
    from scipy import stats as scipy_stats

    parsed      = parse_lavaan(model_syntax)
    structural  = parsed.get("structural", [])
    measurement = parsed.get("measurement", {})

    if not structural:
        raise ValueError("PLSpredict requires a structural model.")

    endogenous = list({r["lhs"] for r in structural})
    preds_by_lhs: dict[str, list[str]] = defaultdict(list)
    for r in structural:
        preds_by_lhs[r["lhs"]].append(r["rhs"])

    composites = _build_composites(df, measurement, structural)

    rng = np.random.default_rng(seed)
    n   = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)

    plspredict_entries: list[PLSPredictEntry] = []
    cvpat_entries: list[CVPATResult] = []

    for lv in endogenous:
        if lv not in composites:
            continue
        predictors = [p for p in preds_by_lhs[lv] if p in composites]
        if not predictors:
            continue

        # Indicators for this endogenous LV
        indicators = [i for i in measurement.get(lv, []) if i in df.columns]
        if not indicators:
            indicators = [lv] if lv in df.columns else []
        if not indicators:
            continue

        y_comp  = composites[lv].values.astype(float)
        X_preds = np.column_stack([composites[p].values for p in predictors])

        # Per-indicator storage: {ind: (model_sq_errors, lm_sq_errors, loss_diffs)}
        ind_model_errs: dict[str, list[float]] = {i: [] for i in indicators}
        ind_lm_errs:    dict[str, list[float]] = {i: [] for i in indicators}
        lv_loss_diffs:  list[float] = []    # for CVPAT (LM - model loss per obs)

        for fold_idx in range(k_folds):
            test_idx  = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_idx])
            if len(train_idx) < len(predictors) + 2:
                continue

            # Train composite model
            y_tr  = y_comp[train_idx]
            X_tr  = np.column_stack([np.ones(len(train_idx)), X_preds[train_idx]])
            X_te  = np.column_stack([np.ones(len(test_idx)),  X_preds[test_idx]])
            try:
                beta_model = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

            # LM baseline: predict each indicator from exogenous composites directly
            for ind in indicators:
                y_ind_tr = df[ind].values[train_idx].astype(float)
                y_ind_te = df[ind].values[test_idx].astype(float)

                # Model: predict composite then scale to indicator
                y_comp_te = X_te @ beta_model
                # Scale factor: OLS of indicator on composite (training)
                y_comp_tr = X_tr @ beta_model
                try:
                    sf = np.linalg.lstsq(
                        np.column_stack([np.ones(len(y_comp_tr)), y_comp_tr]),
                        y_ind_tr, rcond=None
                    )[0]
                    y_model_pred = sf[0] + sf[1] * y_comp_te
                except Exception as _e:
                    y_model_pred = y_comp_te

                # LM baseline: direct OLS from exogenous composites to indicator
                try:
                    beta_lm = np.linalg.lstsq(X_tr, y_ind_tr, rcond=None)[0]
                    y_lm_pred = X_te @ beta_lm
                except Exception as _e:
                    y_lm_pred = np.full(len(test_idx), np.mean(y_ind_tr))

                model_sq = (y_ind_te - y_model_pred) ** 2
                lm_sq    = (y_ind_te - y_lm_pred)    ** 2

                ind_model_errs[ind].extend(model_sq.tolist())
                ind_lm_errs[ind].extend(lm_sq.tolist())
                lv_loss_diffs.extend((lm_sq - model_sq).tolist())

        # PLSpredict entries per indicator
        for ind in indicators:
            me = np.array(ind_model_errs[ind])
            le = np.array(ind_lm_errs[ind])
            if len(me) == 0:
                continue
            rmse_m = float(np.sqrt(np.mean(me)))
            rmse_l = float(np.sqrt(np.mean(le)))
            mae_m  = float(np.mean(np.sqrt(me)))
            mae_l  = float(np.mean(np.sqrt(le)))
            q2p    = _safe_float(1.0 - (rmse_m ** 2 / rmse_l ** 2)) if rmse_l > 0 else None
            plspredict_entries.append(PLSPredictEntry(
                lv=lv, indicator=ind,
                rmse_model=round(rmse_m, 6),
                rmse_lm=round(rmse_l, 6),
                mae_model=round(mae_m, 6),
                mae_lm=round(mae_l, 6),
                q2_predict=round(q2p, 6) if q2p is not None else 0.0,
                better_than_lm=(rmse_m < rmse_l),
            ))

        # CVPAT: one-sample t-test on (LM_loss - model_loss)
        if lv_loss_diffs:
            diffs = np.array(lv_loss_diffs)
            mean_diff = float(np.mean(diffs))
            try:
                t_stat, p_val = scipy_stats.ttest_1samp(diffs, popmean=0)
                p_val = _safe_float(float(p_val))
            except Exception as _e:
                p_val = None
            cvpat_entries.append(CVPATResult(
                lv=lv,
                cvpat_statistic=round(mean_diff, 6),
                p_value=p_val,
                significant=(
                    p_val is not None and
                    p_val < 0.05 and
                    mean_diff > 1e-6   # guard against floating-point near-zero
                ),
                n_folds=k_folds,
            ))

    return plspredict_entries, cvpat_entries


def compute_predict(
    df: pd.DataFrame,
    model_syntax: str,
    omission_distance: int = 7,
    k_folds: int = 10,
    seed: int = 42,
) -> PredictResult:
    """
    Full v0.5 predictive relevance suite:
      - Q² (blindfolding)
      - PLSpredict (k-fold RMSE vs LM baseline)
      - CVPAT (model vs LM loss test)
    """
    q2     = compute_q2(df, model_syntax, omission_distance=omission_distance)
    pls, cvpat = compute_plspredict(df, model_syntax, k_folds=k_folds, seed=seed)
    return PredictResult(q2=q2, plspredict=pls or None, cvpat=cvpat or None)


# ── Code Export ───────────────────────────────────────────────────────────────

def export_as_code(model_syntax: str, algorithm: str = "pls", format: str = "r") -> str:
    parsed = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    structural = parsed.get("structural", [])

    lines = []
    if measurement:
        lines.append("  # Measurement model")
        for lv, indicators in measurement.items():
            lines.append(f"  {lv} =~ {' + '.join(indicators)}")
    if structural:
        lines.append("")
        lines.append("  # Structural model")
        for rel in structural:
            lhs = rel["lhs"]
            rhs_list = rel["rhs"] if isinstance(rel["rhs"], list) else [rel["rhs"]]
            lines.append(f"  {lhs} ~ {' + '.join(rhs_list)}")
    model_block = "\n".join(lines)

    if format == "r":
        estimator_map = {"pls": "ML", "cb": "ML", "wls": "WLS"}
        estimator = estimator_map.get(algorithm, "ML")
        pls_note = (
            "# Note: lavaan does not support PLS-SEM natively.\n"
            "# For PLS in R, consider the 'seminr' package instead.\n"
            "# The syntax below uses lavaan ML as the closest equivalent.\n\n"
        ) if algorithm == "pls" else ""
        return (
            f"# Generated by NAVAL-SEM v0.2.0\n"
            f"# https://github.com/navalsingh9/naval-sem\n\n"
            f"{pls_note}"
            f"library(lavaan)\n\n"
            f"model <- '\n{model_block}\n'\n\n"
            f"fit <- sem(\n"
            f"  model     = model,\n"
            f"  data      = your_data,   # replace with your data frame\n"
            f"  estimator = \"{estimator}\"\n"
            f")\n\n"
            f"summary(fit, fit.measures = TRUE, standardized = TRUE)\n"
            f"fitMeasures(fit, c(\"cfi\", \"rmsea\", \"srmr\", \"aic\", \"bic\"))\n"
        )
    elif format == "python":
        cls = "PLS" if algorithm == "pls" else "Model"
        estimator_line = ", estimator='WLS'" if algorithm == "wls" else ""
        return (
            f"# Generated by NAVAL-SEM v0.2.0\n"
            f"# https://github.com/navalsingh9/naval-sem\n\n"
            f"import pandas as pd\n"
            f"from semopy import {cls}\n\n"
            f"model_syntax = \"\"\"\n{model_block}\n\"\"\"\n\n"
            f"data = pd.read_csv(\"your_data.csv\")  # replace with your file\n"
            f"data = data.dropna()\n\n"
            f"m = {cls}(model_syntax)\n"
            f"m.fit(data{estimator_line})\n\n"
            f"print(m.inspect())\n"
            f"print(m.calc_stats())\n"
        )
    elif format == "lav":
        return (
            f"# NAVAL-SEM export - lavaan syntax\n"
            f"# Compatible with: JASP, jamovi (jSEM module), R lavaan\n"
            f"# Generated by NAVAL-SEM v0.2.0\n"
            f"# https://github.com/navalsingh9/naval-sem\n\n"
            f"{model_block}\n"
        )
    else:
        raise ValueError(f"Unknown export format: '{format}'. Use 'r', 'python', or 'lav'.")
