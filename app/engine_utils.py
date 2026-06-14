"""
engine_utils.py  —  NAVAL-SEM shared primitives
================================================
Low-level helpers used across all engine modules.
Extracted from engine.py so satellites never import private
symbols from the core fitting layer.

Public (internal) API
---------------------
  _emit(log_fn, level, msg)
  _safe_float(val, default)
  _p_to_sig(p)
  _build_composites(df, measurement, structural)
  _build_coef_map(parameters)

Import pattern for all engine satellites:
    from app.engine_utils import _emit, _safe_float, _p_to_sig, _build_composites, _build_coef_map
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("naval_sem.engine")


def _emit(log_fn: Optional[Callable], level: str, msg: str) -> None:
    """Emit a structured log entry via callback and standard logger."""
    if log_fn is not None:
        log_fn(level, msg)
    getattr(logger, level.lower(), logger.info)(msg)


def _safe_float(val, default=None, precision: int = 6):
    """
    Safely coerce val to a rounded float.

    Returns ``default`` (not None) when the value is NaN, Inf, or
    cannot be converted.  Handles pandas Series by extracting the
    first scalar element before conversion.

    Parameters
    ----------
    precision : int
        Decimal places passed to ``round()``.  Default 6.  Pass 12 for
        p-values to prevent tiny values such as 1.8e-9 from rounding to 0.0.
    """
    try:
        if hasattr(val, "iloc"):   # pandas Series — extract scalar first
            if len(val) > 1:
                logger.debug(
                    "_safe_float received a %d-element Series; using iloc[0]. "
                    "This may indicate a logic error upstream.", len(val)
                )
            val = val.iloc[0]
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else round(v, precision)
    except Exception:
        return default


def _p_to_sig(p: Optional[float]) -> bool:
    """Return True when p is not None and p < 0.05."""
    if p is None:
        return False
    return p < 0.05


def _build_composites(
    df: pd.DataFrame,
    measurement: dict[str, list[str]],
    structural: list[dict],
) -> dict[str, pd.Series]:
    """
    Build LV composite scores (mean of indicators) for all model variables.

    For latent variables: composite = unweighted mean of their indicators.
    For observed variables in structural paths with no measurement block:
    composite = the column itself.

    Parameters
    ----------
    df          : pd.DataFrame
    measurement : {lv_name: [indicator, ...]}
    structural  : list of {lhs, rhs} dicts from parse_lavaan()

    Returns
    -------
    {name: pd.Series}
    """
    composites: dict[str, pd.Series] = {}

    for lv, indicators in measurement.items():
        cols = [c for c in indicators if c in df.columns]
        if cols:
            composites[lv] = df[cols].mean(axis=1)

    for rel in structural:
        for var in (rel["lhs"], rel["rhs"]):
            if var not in composites and var in df.columns:
                composites[var] = df[var]

    return composites


def _build_squared_terms(df: pd.DataFrame, composites: dict, nonlinear_terms: list) -> pd.DataFrame:
    """
    For each nonlinear term, mean-centre the base composite and create
    a squared column. Returns augmented DataFrame copy (never mutates df).
    """
    df = df.copy()
    for term in nonlinear_terms:
        base   = term["base_var"]
        sq_col = term["sq_col"]
        series = composites.get(base)
        if series is None and base in df.columns:
            series = df[base].astype(float)
        if series is None:
            continue
        mc = series - series.mean()
        df[sq_col] = mc ** 2
    return df


def _build_coef_map(parameters) -> dict:
    """Return {(rhs, lhs): coef} for structural paths (op == '~').

    Convention: predictor-first (rhs), then outcome (lhs) — consistent
    with the tuple key order used across engine.py, engine_mga.py, and
    engine_moderation.py bootstrap loops.

    Parameters
    ----------
    parameters : list[PathParameter]
        The ``.parameters`` list from a ``SemResult`` / ``ModerationResult``.

    Returns
    -------
    dict  {(rhs_name, lhs_name): float_estimate}
    """
    return {
        (p.rhs, p.lhs): p.estimate
        for p in parameters
        if hasattr(p, "op") and p.op == "~"
    }


def _coef_from_params(parameters, lhs: str, rhs: str):
    """Return the estimate for a specific structural (lhs ~ rhs) path, or None."""
    for p in parameters:
        if getattr(p, 'op', None) == '~' and p.lhs == lhs and p.rhs == rhs:
            return p.estimate
    return None


def _ci_from_bootstrap(samples: list) -> tuple:
    """Return (ci_lo_95, ci_hi_95) from a bootstrap distribution, or (None, None)."""
    if len(samples) < 10:
        return None, None
    import numpy as np
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _sig_from_ci(ci_lo, ci_hi) -> bool:
    """Return True when a 95% CI excludes zero."""
    if ci_lo is None or ci_hi is None:
        return False
    return not (ci_lo <= 0.0 <= ci_hi)
