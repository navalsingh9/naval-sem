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
from typing import Optional

from app.parser import parse_lavaan, build_semopy_syntax
from app.schemas import (
    ModelResult, PathParameter, FitIndices,
    BootstrapResult, HTMTResult, HTMTEntry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        if hasattr(val, 'iloc'):   # pandas Series — extract scalar first
            val = val.iloc[0]
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    except Exception:
        return default


def _p_to_sig(p: Optional[float]) -> bool:
    if p is None:
        return False
    return p < 0.05


def _fit_verdict(fit: FitIndices) -> FitIndices:
    if fit.cfi is not None:
        fit.cfi_acceptable = fit.cfi >= 0.90
        fit.cfi_good = fit.cfi >= 0.95
    if fit.rmsea is not None:
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

    # ── Stage 2: standardize if any |loading| > 1 and data is available ─────
    needs_std = any(abs(l) > 1.0 for lams in loadings.values() for l in lams)
    if needs_std and df is not None:
        std_loadings: dict[str, list[float]] = {}
        for lv, indicators in measurement.items():
            cols = [c for c in indicators if c in df.columns]
            if not cols:
                std_loadings[lv] = loadings.get(lv, [])
                continue
            # Construct composite = unweighted mean of available indicators
            composite = df[cols].mean(axis=1)
            comp_std  = composite.std()
            if comp_std == 0:
                std_loadings[lv] = loadings.get(lv, [])
                continue
            lam_std = []
            for ind in indicators:
                if ind not in df.columns:
                    continue
                ind_std = df[ind].std()
                if ind_std == 0:
                    continue
                r = float(df[ind].corr(composite))
                val = _safe_float(r)
                if val is not None:
                    lam_std.append(val)
            if lam_std:
                std_loadings[lv] = lam_std
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
            except Exception:
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
        except Exception:
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
    except Exception:
        return None



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
        except Exception:
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


# ── Main fit function ─────────────────────────────────────────────────────────

def fit_model(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
) -> ModelResult:
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed. Run: pip install semopy")

    parsed = parse_lavaan(model_syntax)
    syntax = build_semopy_syntax(parsed)
    warnings = []

    missing_cols = [v for v in parsed["observed_vars"] if v not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found in data: {missing_cols}. "
            f"Available: {df.columns.tolist()}"
        )

    estimator = "ML"
    use_pls = False
    if algorithm == "pls":
        try:
            from semopy import PLS
            use_pls = True
        except ImportError:
            warnings.append(
                "semopy.PLS not available — falling back to CB-SEM (ML)."
            )
    elif algorithm == "wls":
        estimator = "WLS"

    try:
        if use_pls:
            from semopy import PLS
            sem_model = PLS(syntax)
            try:
                sem_model.fit(df)
            except TypeError:
                warnings.append("PLS not supported in this semopy version — using CB-SEM (ML).")
                sem_model = Model(syntax)
                sem_model.fit(df, estimator="ML")
            algo_label = "PLS-SEM"
        else:
            sem_model = Model(syntax)
            if estimator == "WLS":
                sem_model.fit(df, obj="WLS")
            else:
                sem_model.fit(df)
            algo_label = "CB-SEM" if algorithm == "cb" else ("WLS" if algorithm == "wls" else "CB-SEM (ML)")
    except Exception as e:
        raise ValueError(f"Model did not converge: {e}")

    params_df = sem_model.inspect()

    parameters = []
    for _, row in params_df.iterrows():
        est = _safe_float(row.get("Estimate", row.get("estimate", 0.0)), 0.0)
        se = _safe_float(row.get("Std. Err.", row.get("std_err", None)))
        z = _safe_float(row.get("z-Value", row.get("z_value", None)))
        p = _safe_float(row.get("p-Value", row.get("p_value", None)))
        ci_lo = round(est - 1.96 * se, 6) if se is not None else None
        ci_hi = round(est + 1.96 * se, 6) if se is not None else None
        lhs = str(row.get("lval", row.get("lhs", "")))
        op = str(row.get("op", "~"))
        rhs = str(row.get("rval", row.get("rhs", "")))
        parameters.append(PathParameter(
            lhs=lhs, op=op, rhs=rhs,
            estimate=est,
            std_error=se if se is not None else 0.0,
            z_value=z if z is not None else 0.0,
            p_value=p if p is not None else 1.0,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            significant=_p_to_sig(p),
        ))

    fit = FitIndices()

    # ── Standard fit statistics ───────────────────────────────────────────────
    try:
        from semopy import calc_stats
        stats = calc_stats(sem_model)

        def gs(key):
            for k, v in stats.items():
                if k.lower().replace(" ", "_") == key.lower().replace(" ", "_"):
                    return _safe_float(v)
            return None

        fit.cfi = gs("CFI")
        fit.rmsea = gs("RMSEA")
        fit.srmr = gs("SRMR")
        if fit.srmr is None:
            fit.srmr = _compute_srmr(sem_model, df, params_df, parsed)
        fit.chi_square = gs("chi2") or gs("chi_square")
        fit.df = int(gs("df") or 0) or None
        fit.p_value = gs("p-value") or gs("p_value")
        fit.aic = gs("AIC")
        fit.bic = gs("BIC")

        r2 = {}
        for rel in parsed["structural"]:
            lhs = rel["lhs"]
            try:
                pred = sem_model.predict(df)
                if lhs in pred.columns and lhs in df.columns:
                    ss_res = ((df[lhs] - pred[lhs]) ** 2).sum()
                    ss_tot = ((df[lhs] - df[lhs].mean()) ** 2).sum()
                    r2[lhs] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
            except Exception:
                pass
        if r2:
            fit.r_squared = r2

    except Exception as e:
        warnings.append(f"Could not compute fit statistics: {e}")

    # ── Measurement validity metrics ──────────────────────────────────────────
    measurement = parsed.get("measurement", {})

    loadings: dict = {}
    try:
        loadings = _extract_loadings(params_df, measurement, df)
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

    return ModelResult(
        algorithm=algo_label,
        n_obs=len(df),
        n_params=len(parameters),
        converged=True,
        parameters=parameters,
        fit=fit,
        latent_variables=parsed["latent_vars"],
        observed_variables=parsed["observed_vars"],
        warnings=warnings,
    )


# ── Bootstrapping ─────────────────────────────────────────────────────────────

def run_bootstrap(
    df: pd.DataFrame,
    model_syntax: str,
    n: int = 500,
    algorithm: str = "pls",
    seed: int = 42,
) -> BootstrapResult:
    try:
        from semopy import Model
    except ImportError:
        raise RuntimeError("semopy is not installed.")

    parsed = parse_lavaan(model_syntax)
    syntax = build_semopy_syntax(parsed)
    rng = np.random.default_rng(seed)

    all_estimates = []
    converged = 0

    for _ in range(n):
        sample = df.sample(frac=1, replace=True, random_state=int(rng.integers(1e6)))
        try:
            m = Model(syntax)
            m.fit(sample)
            p = m.inspect()
            row_vals = p["Estimate"].values if "Estimate" in p.columns else p["estimate"].values
            all_estimates.append(row_vals)
            converged += 1
        except Exception:
            continue

    if not all_estimates:
        raise ValueError("No bootstrap samples converged.")

    est_array = np.array(all_estimates)
    bs_se = np.std(est_array, axis=0, ddof=1)
    ci_lo = np.percentile(est_array, 2.5, axis=0)
    ci_hi = np.percentile(est_array, 97.5, axis=0)
    bs_mean = np.mean(est_array, axis=0)

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
    except Exception:
        labels = [{"lhs": f"param_{i}", "op": "~", "rhs": ""} for i in range(len(bs_se))]

    parameters = []
    for i, lab in enumerate(labels):
        if i >= len(bs_se):
            break
        parameters.append({
            **lab,
            "bs_mean": round(float(bs_mean[i]), 6),
            "bs_se": round(float(bs_se[i]), 6),
            "ci_lower_95": round(float(ci_lo[i]), 6),
            "ci_upper_95": round(float(ci_hi[i]), 6),
            "significant": not (ci_lo[i] <= 0 <= ci_hi[i]),
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
            cross = mean_abs_corr(inds_a, inds_b)
            within_a = mean_abs_corr(inds_a, inds_a, same=True)
            within_b = mean_abs_corr(inds_b, inds_b, same=True)
            denom = np.sqrt(within_a * within_b)
            htmt_val = cross / denom if denom > 0 else np.nan
            entries.append(HTMTEntry(
                construct_a=lv_a,
                construct_b=lv_b,
                htmt=round(float(htmt_val), 4) if not np.isnan(htmt_val) else 9999.0,
                acceptable=htmt_val < 0.90 if not np.isnan(htmt_val) else False,
            ))

    return HTMTResult(
        matrix=entries,
        all_acceptable=all(e.acceptable for e in entries),
    )


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
