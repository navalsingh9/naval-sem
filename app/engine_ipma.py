"""
engine_ipma.py  —  NAVAL-SEM v0.7
===================================
Importance-Performance Map Analysis (IPMA).

Public API
----------
  compute_ipma(df, model_syntax, target_lv, algorithm,
               scale_min, scale_max, log_fn) -> IPMAResult

Reference
---------
  Ringle & Sarstedt (2016); Hair et al. (2022, Chapter 7).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _emit, _safe_float
from app.engine import compute_indirect_effects, fit_model
from app.parser import parse_lavaan
from app.schemas import IPMAEntry, IPMAResult, PathParameter

logger = logging.getLogger("naval_sem.ipma")


# ── private helper ─────────────────────────────────────────────────────────────

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


# ── public API ─────────────────────────────────────────────────────────────────

def compute_ipma(
    df: pd.DataFrame,
    model_syntax: str,
    target_lv: str,
    algorithm: str = "pls",
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    log_fn: Optional[Callable] = None,
) -> IPMAResult:
    """
    Importance-Performance Map Analysis (IPMA).

    Importance  = total effect of each predictor on ``target_lv``
                  (direct path + all indirect paths combined).
    Performance = mean of the predictor's composite score, rescaled to 0–100
                  using the theoretical scale range [scale_min, scale_max].
                  If ``scale_min`` / ``scale_max`` are None, the observed
                  minimum and maximum of the composite are used.

    Parameters
    ----------
    df           : pd.DataFrame
    model_syntax : str     lavaan syntax.
    target_lv    : str     The dependent LV for which importance is computed.
    algorithm    : str     ``"pls"`` | ``"cb"`` | ``"wls"``.
    scale_min    : float   Theoretical scale minimum (e.g. 1 for Likert 1–5).
    scale_max    : float   Theoretical scale maximum (e.g. 5 for Likert 1–5).
    log_fn       : callable | None

    Returns
    -------
    IPMAResult
        Entries sorted by importance (descending).
    """
    _emit(log_fn, "step", f"IPMA: target LV = '{target_lv}'")

    parsed      = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    warnings:   list[str] = []

    if target_lv not in parsed.get("latent_vars", []) + parsed.get("observed_vars", []):
        raise ValueError(
            f"IPMA: target LV '{target_lv}' not found in the model. "
            f"Available: {parsed.get('latent_vars', [])}"
        )

    # ── Fit model to get total effects ────────────────────────────────────────
    _emit(log_fn, "step", "IPMA: fitting model to extract total effects")
    try:
        res = fit_model(df, model_syntax, algorithm=algorithm,
                        bootstrap_n=0, log_fn=None)
    except Exception as exc:
        raise ValueError(f"IPMA: model fit failed — {exc}") from exc

    try:
        indirect_res  = compute_indirect_effects(
            df, model_syntax, 
            n_bootstrap=0, log_fn=None,
        )
        total_effects = indirect_res.total_effects   # {from: {to: float}}
    except Exception as exc:
        warnings.append(
            f"IPMA: indirect effects computation failed ({exc}). "
            "Using direct effects only."
        )
        total_effects = {}
        for p in res.parameters:
            if p.op == "~" and p.lhs == target_lv:
                total_effects.setdefault(p.rhs, {})[target_lv] = p.estimate

    # Predictors: LVs / observed vars that have a total effect on target_lv
    predictors = [
        lv for lv, targets in total_effects.items()
        if target_lv in targets and lv != target_lv
    ]
    if not predictors:
        predictors = [
            p.rhs for p in res.parameters
            if p.op == "~" and p.lhs == target_lv
        ]
        warnings.append("IPMA: no total-effect data; using direct paths only.")

    if not predictors:
        raise ValueError(
            f"IPMA: no predictors of '{target_lv}' found in model."
        )

    # ── Composite scores ───────────────────────────────────────────────────────
    _emit(log_fn, "step", "IPMA: computing composite scores")
    composites = _build_composites(df, measurement, parsed.get("structural", []))

    # ── Scale range ────────────────────────────────────────────────────────────
    if scale_min is None or scale_max is None:
        all_ind_vals: list[float] = []
        for lv in predictors:
            for ind in measurement.get(lv, []):
                if ind in df.columns:
                    all_ind_vals.extend(df[ind].dropna().tolist())
        if all_ind_vals:
            obs_min = float(np.min(all_ind_vals))
            obs_max = float(np.max(all_ind_vals))
        else:
            obs_min, obs_max = 1.0, 5.0   # default Likert 1–5

        eff_min = scale_min if scale_min is not None else obs_min
        eff_max = scale_max if scale_max is not None else obs_max
        if scale_min is None or scale_max is None:
            warnings.append(
                f"IPMA: scale range not provided — using observed range "
                f"[{eff_min:.2f}, {eff_max:.2f}]. Pass scale_min / scale_max "
                "for correct 0–100 rescaling."
            )
    else:
        eff_min, eff_max = scale_min, scale_max

    scale_range = eff_max - eff_min
    if scale_range < 1e-12:
        scale_range = 1.0
        warnings.append("IPMA: scale_min == scale_max; performance set to 50.")

    # ── Build entries ──────────────────────────────────────────────────────────
    entries: list[IPMAEntry] = []

    for lv in predictors:
        importance = _safe_float(total_effects.get(lv, {}).get(target_lv))
        if importance is None:
            importance = _safe_float(
                _coef_from_params(res.parameters, target_lv, lv)
            ) or 0.0

        comp = composites.get(lv)
        if comp is None and lv in df.columns:
            comp = df[lv].astype(float)

        if comp is not None:
            raw_mean = float(comp.mean())
        else:
            warnings.append(
                f"IPMA: no composite data for '{lv}'; performance set to 50."
            )
            raw_mean = (eff_min + eff_max) / 2

        performance = round((raw_mean - eff_min) / scale_range * 100, 2)
        performance = max(0.0, min(100.0, performance))

        entries.append(IPMAEntry(
            lv=lv,
            importance=round(float(importance), 6),
            performance=performance,
        ))

    entries.sort(key=lambda e: e.importance, reverse=True)

    _emit(log_fn, "ok",
          f"IPMA complete — {len(entries)} predictors of '{target_lv}'")

    return IPMAResult(
        target_lv=target_lv,
        entries=entries,
        scale_min=eff_min,
        scale_max=eff_max,
        algorithm=algorithm,
        warnings=warnings,
    )
