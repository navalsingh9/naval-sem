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
    VIFEntry, F2Entry, IndirectEffect, IndirectResult, OuterWeightEntry,
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
    bootstrap_n: int = 0,
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

    outer_weight_entries: list[OuterWeightEntry] = []
    if bootstrap_n > 0:
        try:
            outer_weight_entries = compute_outer_weight_significance(
                df, model_syntax, n=bootstrap_n
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
    missing_pvals = any(p.p_value >= 0.999 for p in structural_params)

    if bootstrap_n > 0 and (missing_pvals or use_pls):
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


# ── Outer weight significance ─────────────────────────────────────────────────

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
        except Exception:
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
            except Exception:
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
    except Exception:
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
        except Exception:
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
        except Exception:
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
            except Exception:
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
