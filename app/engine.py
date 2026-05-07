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
    return fit


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
