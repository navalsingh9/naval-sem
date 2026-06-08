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

Import pattern for all engine satellites:
    from app.engine_utils import _emit, _safe_float, _p_to_sig, _build_composites
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


def _safe_float(val, default=None):
    """
    Safely coerce val to a rounded float.

    Returns ``default`` (not None) when the value is NaN, Inf, or
    cannot be converted.  Handles pandas Series by extracting the
    first scalar element before conversion.
    """
    try:
        if hasattr(val, "iloc"):   # pandas Series — extract scalar first
            val = val.iloc[0]
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else round(v, 6)
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
