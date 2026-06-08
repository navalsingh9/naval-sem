"""
engine_moderation.py  —  NAVAL-SEM v0.7
========================================
Moderation analysis via the product-of-composites approach.

Public API
----------
  run_moderation(df, model_syntax, algorithm, bootstrap_n, seed, log_fn)
    -> ModerationResult

All functions follow engine.py conventions:
  - _safe_float(), _emit() from engine_utils
  - warnings.append() for non-fatal issues
  - try/except on every external call
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _emit, _safe_float
from app.engine import fit_model
from app.parser import (
    build_semopy_syntax,
    detect_interactions,
    expand_interaction_terms,
    parse_lavaan,
)
from app.schemas import (
    FitIndices,
    ModerationResult,
    ModerationTerm,
    PathParameter,
    SimpleSlope,
)

logger = logging.getLogger("naval_sem.moderation")


# ── private helpers ────────────────────────────────────────────────────────────

def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R² of y on X (with intercept). Returns 0.0 on any failure."""
    try:
        X_aug = np.column_stack([np.ones(len(X)), X])
        coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat  = X_aug @ coefs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-14:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)
    except Exception:
        return 0.0


def _coef_from_params(
    parameters: list[PathParameter],
    lhs: str,
    rhs: str,
) -> Optional[float]:
    """Return the estimate for a specific (lhs ~ rhs) path, or None."""
    for p in parameters:
        if p.op == "~" and p.lhs == lhs and p.rhs == rhs:
            return p.estimate
    return None


def _ci_from_bootstrap(
    samples: list[float],
) -> tuple[Optional[float], Optional[float]]:
    """Return (ci_lo_95, ci_hi_95) from a bootstrap distribution."""
    if len(samples) < 10:
        return None, None
    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


# ── public API ─────────────────────────────────────────────────────────────────

def run_moderation(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
    bootstrap_n: int = 500,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> ModerationResult:
    """
    Moderation analysis via the product-of-composites approach.

    Detects ``X*M`` interaction terms in the lavaan structural syntax and
    tests each one as follows:

    1. Compute composite scores for X (IV) and M (moderator).
    2. Mean-centre both composites.
    3. Create the product column ``X_x_M = X_mc × M_mc``.
    4. Fit the full model (with the interaction term) using ``fit_model()``.
    5. Compute Δ R² and Cohen's f² by comparing R² with vs. without the
       interaction term.
    6. Bootstrap simple slopes at moderator = −1 SD, mean, +1 SD.

    Syntax example::

        Y  ~  X + M + X*M
        X  =~ x1 + x2 + x3
        M  =~ m1 + m2 + m3
        Y  =~ y1 + y2 + y3

    Parameters
    ----------
    df           : pd.DataFrame
    model_syntax : str   lavaan syntax containing at least one ``X*M`` term.
    algorithm    : str   ``"pls"`` (default) | ``"cb"`` | ``"wls"``.
    bootstrap_n  : int   Bootstrap resamples for simple-slope CIs.
    seed         : int
    log_fn       : callable | None

    Returns
    -------
    ModerationResult
    """
    _emit(log_fn, "step", "Moderation: parsing syntax and detecting interaction terms")

    parsed_orig  = parse_lavaan(model_syntax)
    interactions = detect_interactions(parsed_orig)

    if not interactions:
        raise ValueError(
            "No interaction terms detected in model syntax. "
            "Use 'X*M' notation in a structural path to specify moderation, "
            "e.g.  Y ~ X + M + X*M"
        )

    _emit(log_fn, "info",
          f"  {len(interactions)} interaction term(s): "
          + ", ".join(f"{i['iv']}*{i['moderator']}" for i in interactions))

    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    # ── Expand: create product columns, patch parsed dict ─────────────────────
    try:
        parsed_full, df_aug = expand_interaction_terms(parsed_orig, df)
    except ValueError as exc:
        raise ValueError(f"Moderation setup failed: {exc}") from exc

    syntax_full = build_semopy_syntax(parsed_full)

    # ── Fit full model (with interaction) ─────────────────────────────────────
    _emit(log_fn, "step", "Moderation: fitting full model (with interaction term)")
    try:
        res_full = fit_model(
            df_aug, syntax_full,
            algorithm=algorithm, bootstrap_n=0, log_fn=None,
        )
    except Exception as exc:
        raise ValueError(f"Moderation full-model fit failed: {exc}") from exc

    # ── Build one ModerationTerm per interaction ───────────────────────────────
    moderation_terms: list[ModerationTerm] = []

    for itx in interactions:
        iv      = itx["iv"]
        mod     = itx["moderator"]
        outcome = itx["outcome"]
        icol    = itx["interaction_col"]

        _emit(log_fn, "step", f"  Moderation: {iv} × {mod} → {outcome}")

        # Path coefficients from full model
        beta_iv  = _safe_float(_coef_from_params(res_full.parameters, outcome, iv))  or 0.0
        beta_mod = _safe_float(_coef_from_params(res_full.parameters, outcome, mod)) or 0.0
        beta_int = _safe_float(_coef_from_params(res_full.parameters, outcome, icol)) or 0.0

        # R² with interaction
        r2_with = _safe_float((res_full.fit.r_squared or {}).get(outcome)) or 0.0

        # R² without interaction: refit with interaction term removed
        parsed_reduced = copy.deepcopy(parsed_full)
        parsed_reduced["structural"] = [
            r for r in parsed_reduced["structural"]
            if not (r["lhs"] == outcome and r["rhs"] == icol)
        ]
        syntax_reduced = build_semopy_syntax(parsed_reduced)

        try:
            res_red    = fit_model(df_aug, syntax_reduced,
                                   algorithm=algorithm, bootstrap_n=0, log_fn=None)
            r2_without = _safe_float((res_red.fit.r_squared or {}).get(outcome)) or 0.0
        except Exception as exc:
            warnings.append(
                f"Reduced model fit failed for {iv}×{mod}: {exc}. "
                "Δ R² and f² set to 0."
            )
            r2_without = r2_with

        delta_r2 = max(0.0, r2_with - r2_without)
        denom    = max(1.0 - r2_with, 1e-12)
        f2_int   = round(delta_r2 / denom, 6)

        # ── Bootstrap CIs for β_interaction and simple slopes ─────────────────
        _emit(log_fn, "step",
              f"  Bootstrap CIs ({bootstrap_n} samples) for {iv}×{mod}")

        bs_beta_int:   list[float] = []
        bs_slopes_lo:  list[float] = []
        bs_slopes_mid: list[float] = []
        bs_slopes_hi:  list[float] = []

        mod_composite = _build_composites(
            df,
            parsed_orig.get("measurement", {}),
            parsed_orig.get("structural", []),
        ).get(mod)
        if mod_composite is None and mod in df.columns:
            mod_composite = df[mod].astype(float)

        mod_sd     = float(mod_composite.std()) if mod_composite is not None else 1.0
        mod_levels = [-mod_sd, 0.0, mod_sd]

        for _bi in range(bootstrap_n):
            try:
                s_idx = rng.integers(0, len(df_aug), size=len(df_aug))
                df_bs = df_aug.iloc[s_idx].reset_index(drop=True)
                r_bs  = fit_model(df_bs, syntax_full,
                                  algorithm=algorithm, bootstrap_n=0)
                b_x   = _coef_from_params(r_bs.parameters, outcome, iv)
                b_int = _coef_from_params(r_bs.parameters, outcome, icol)
                if b_x is None or b_int is None:
                    continue
                bs_beta_int.append(b_int)
                for lvl, store in zip(mod_levels,
                                      [bs_slopes_lo, bs_slopes_mid, bs_slopes_hi]):
                    store.append(b_x + b_int * lvl)
            except Exception:
                continue

        ci_lo, ci_hi = _ci_from_bootstrap(bs_beta_int)

        # Simple slopes
        level_labels  = ["low (−1 SD)", "mean (0)", "high (+1 SD)"]
        simple_slopes: list[SimpleSlope] = []
        for label, lvl, bs_store in zip(
            level_labels, mod_levels,
            [bs_slopes_lo, bs_slopes_mid, bs_slopes_hi]
        ):
            slope_obs    = beta_iv + beta_int * lvl
            s_lo, s_hi   = _ci_from_bootstrap(bs_store)
            sig = (not (s_lo <= 0.0 <= s_hi)) if (s_lo is not None and s_hi is not None) else False
            simple_slopes.append(SimpleSlope(
                moderator_level=label,
                moderator_value=round(lvl, 4),
                slope=round(slope_obs, 6),
                ci_lower_95=round(s_lo, 6) if s_lo is not None else None,
                ci_upper_95=round(s_hi, 6) if s_hi is not None else None,
                significant=sig,
            ))

        moderation_terms.append(ModerationTerm(
            iv=iv, moderator=mod, outcome=outcome,
            interaction_col=icol,
            beta_iv=round(beta_iv, 6),
            beta_moderator=round(beta_mod, 6),
            beta_interaction=round(beta_int, 6),
            ci_lower_95=round(ci_lo, 6) if ci_lo is not None else None,
            ci_upper_95=round(ci_hi, 6) if ci_hi is not None else None,
            significant=(not (ci_lo <= 0.0 <= ci_hi))
                        if (ci_lo is not None and ci_hi is not None) else False,
            r2_with=round(r2_with, 6),
            r2_without=round(r2_without, 6),
            delta_r2=round(delta_r2, 6),
            f2_interaction=f2_int,
            simple_slopes=simple_slopes,
        ))

    if not moderation_terms:
        raise ValueError("Moderation: no terms could be estimated.")

    _emit(log_fn, "ok",
          f"Moderation complete — {len(moderation_terms)} term(s) estimated")

    return ModerationResult(
        algorithm=algorithm,
        n_obs=res_full.n_obs,
        bootstrap_n=bootstrap_n,
        moderation_terms=moderation_terms,
        parameters=res_full.parameters,
        fit=res_full.fit,
        warnings=warnings + res_full.warnings,
    )
