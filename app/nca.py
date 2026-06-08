"""
nca.py  —  NAVAL-SEM v0.7
==========================
Necessary Condition Analysis (NCA).
Dul, J. (2016). Necessary Condition Analysis (NCA).
Organizational Research Methods, 19(1), 10-52.

Public API
----------
  compute_nca(df, model_syntax, n_permutations, seed, log_fn) -> NCAResult

Internal helpers
----------------
  _ce_fdh(x, y)   : CE-FDH ceiling line and effect size d
  _cr_fdh(x, y)   : CR-FDH ceiling line and effect size d
  _nca_label(d)   : effect size label
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── Top-level imports (moved from inside compute_nca to fix TC-26 / TC-49) ────
from app.engine_utils import _build_composites, _emit, _safe_float
from app.parser import parse_lavaan
from app.schemas import NCAEntry, NCAResult

logger = logging.getLogger("naval_sem.nca")


# ── Effect size label ──────────────────────────────────────────────────────────

def _nca_label(d: float) -> str:
    """
    Map NCA effect size d to a descriptive label (Dul 2016 benchmarks).

    =========  ==============
    d          Label
    =========  ==============
    < 0.1      negligible
    0.1–0.3    small
    0.3–0.5    medium
    ≥ 0.5      large
    =========  ==============
    """
    if d < 0.1:
        return "negligible"
    elif d < 0.3:
        return "small"
    elif d < 0.5:
        return "medium"
    return "large"


# ── CE-FDH ceiling ─────────────────────────────────────────────────────────────

def _ce_fdh(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    """
    CE-FDH: Ceiling Envelopment — Free Disposal Hull.

    Constructs the upper-left staircase boundary of the scatter plot.
    Effect size d = ceiling zone / scope.

    Algorithm
    ---------
    1. Scan observations right-to-left (descending X).
    2. A point (x_i, y_i) is on the ceiling if no observed point has
       x ≤ x_i and y > y_i  ("not dominated from above-left").
    3. The ceiling is a non-decreasing step function as X increases.
    4. Ceiling zone = area between the ceiling and y_max within the scope.
    5. Scope = (x_max - x_min) × (y_max - y_min).

    Parameters
    ----------
    x, y : np.ndarray (1-D, same length, no NaN)

    Returns
    -------
    (d, ceiling_x, ceiling_y)
        d             Effect size (0–1).
        ceiling_x/y   Staircase ceiling line coordinates for plotting
                      (sorted by x ascending; step function pairs).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    scope = (x_max - x_min) * (y_max - y_min)
    if scope < 1e-14:
        return 0.0, list(x), list(y)

    # Find ceiling observations: scan right-to-left
    order = np.argsort(x)[::-1]   # descending x
    ceil_pts: list[tuple[float, float]] = []
    running_max_y = -np.inf
    for idx in order:
        xi, yi = float(x[idx]), float(y[idx])
        if yi > running_max_y:
            running_max_y = yi
            ceil_pts.append((xi, yi))

    ceil_pts.sort(key=lambda p: p[0])   # ascending x

    # Build step-function coordinates for plotting
    plot_x: list[float] = []
    plot_y: list[float] = []
    if ceil_pts:
        # Start at x_min with the height of the first ceiling point
        # (extend leftward at the first point's y-level)
        plot_x.append(x_min)
        plot_y.append(ceil_pts[0][1])
        for xi, yi in ceil_pts:
            # Vertical step up at this x
            plot_x.append(xi)
            plot_y.append(plot_y[-1])   # carry previous y to this x
            plot_x.append(xi)
            plot_y.append(yi)            # jump up
        # Extend to x_max
        plot_x.append(x_max)
        plot_y.append(ceil_pts[-1][1])

    # Ceiling zone: sum of rectangles above each ceiling step
    ceiling_zone = 0.0
    # Between consecutive ceiling points, ceiling = y of the LEFT point (step fn)
    prev_x = x_min
    prev_y = ceil_pts[0][1] if ceil_pts else y_max

    for xi, yi in ceil_pts:
        # Rectangle from prev_x to xi, height = y_max - prev_y
        width  = xi - prev_x
        height = y_max - prev_y
        if width > 0 and height > 0:
            ceiling_zone += width * height
        prev_x = xi
        prev_y = yi

    # Final rectangle from last ceiling point to x_max
    width  = x_max - prev_x
    height = y_max - prev_y
    if width > 0 and height > 0:
        ceiling_zone += width * height

    d = round(float(ceiling_zone / scope), 6)
    d = max(0.0, min(1.0, d))

    return d, plot_x, plot_y


# ── CR-FDH ceiling ─────────────────────────────────────────────────────────────

def _cr_fdh(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, list[float], list[float]]:
    """
    CR-FDH: Ceiling Regression — Free Disposal Hull.

    Fits an OLS regression line through the CE-FDH ceiling observations.
    Effect size d = ceiling zone (above regression line) / scope.

    Parameters
    ----------
    x, y : np.ndarray

    Returns
    -------
    (d, slope, intercept, line_x, line_y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    scope = (x_max - x_min) * (y_max - y_min)
    if scope < 1e-14:
        return 0.0, 0.0, 0.0, [x_min, x_max], [y_max, y_max]

    # Get CE-FDH ceiling observations (the "upper-left frontier" points)
    _, _, _ = _ce_fdh(x, y)   # re-derive ceiling pts
    order = np.argsort(x)[::-1]
    ceil_xs: list[float] = []
    ceil_ys: list[float] = []
    running_max_y = -np.inf
    for idx in order:
        xi, yi = float(x[idx]), float(y[idx])
        if yi > running_max_y:
            running_max_y = yi
            ceil_xs.append(xi)
            ceil_ys.append(yi)
    ceil_xs = ceil_xs[::-1]
    ceil_ys = ceil_ys[::-1]

    # OLS through ceiling observations
    cx = np.array(ceil_xs)
    cy = np.array(ceil_ys)

    if len(cx) < 2:
        # Degenerate: constant ceiling
        intercept = float(cy[0]) if len(cy) > 0 else y_max
        slope     = 0.0
    else:
        try:
            X_aug = np.column_stack([np.ones(len(cx)), cx])
            coefs, *_ = np.linalg.lstsq(X_aug, cy, rcond=None)
            intercept, slope = float(coefs[0]), float(coefs[1])
        except Exception:
            slope, intercept = 0.0, y_max

    # Regression line at x_min and x_max
    y_line_min = slope * x_min + intercept
    y_line_max = slope * x_max + intercept
    line_x = [x_min, x_max]
    line_y = [y_line_min, y_line_max]

    # Ceiling zone: area above regression line, below y_max, within scope
    # Integral of max(0, y_max - (slope*x + intercept)) from x_min to x_max
    # = (y_max - intercept)(x_max - x_min)  - slope*(x_max² - x_min²)/2
    # Clamped to 0 where regression exceeds y_max
    a = y_max - intercept
    ceiling_zone = a * (x_max - x_min) - slope * (x_max**2 - x_min**2) / 2
    # Clamp (regression line might exceed y_max at some x)
    ceiling_zone = max(0.0, ceiling_zone)

    d = round(float(ceiling_zone / scope), 6)
    d = max(0.0, min(1.0, d))

    return d, round(slope, 6), round(intercept, 6), line_x, line_y


# ── Main NCA function ──────────────────────────────────────────────────────────

def compute_nca(
    df: pd.DataFrame,
    model_syntax: str,
    n_permutations: int = 1000,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> NCAResult:
    """
    Necessary Condition Analysis (NCA) across all structural IV→DV pairs.

    For each structural path ``X → Y`` in the model:

    1. Compute composite scores for X and Y (mean of indicators).
    2. Compute CE-FDH effect size d and ceiling staircase.
    3. Compute CR-FDH effect size d and ceiling regression line.
    4. Permutation test: shuffle Y, recompute d → p-value.

    Parameters
    ----------
    df            : pd.DataFrame
    model_syntax  : str    lavaan syntax.
    n_permutations: int    Permutation samples for significance test.
    seed          : int
    log_fn        : callable | None

    Returns
    -------
    NCAResult
    """
    _emit(log_fn, "step", f"NCA: parsing model ({n_permutations} permutations)")

    parsed      = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    structural  = parsed.get("structural",  [])
    warnings:   list[str] = []

    if not structural:
        raise ValueError("NCA requires at least one structural path.")

    rng = np.random.default_rng(seed)

    # ── Compute all composites once ────────────────────────────────────────────
    composites = _build_composites(df, measurement, structural)

    def _get_scores(name: str) -> Optional[np.ndarray]:
        if name in composites:
            return composites[name].values
        if name in df.columns:
            return df[name].astype(float).values
        return None

    # ── Process each structural pair ───────────────────────────────────────────
    entries: list[NCAEntry] = []
    # Deduplicate paths (keep unique iv→dv pairs)
    seen_pairs: set[tuple[str, str]] = set()

    for rel in structural:
        iv, dv = rel["rhs"], rel["lhs"]
        if (iv, dv) in seen_pairs:
            continue
        seen_pairs.add((iv, dv))

        _emit(log_fn, "info", f"  NCA: {iv} → {dv}")

        x_arr = _get_scores(iv)
        y_arr = _get_scores(dv)

        if x_arr is None:
            warnings.append(f"NCA: no data for IV '{iv}' — skipped.")
            continue
        if y_arr is None:
            warnings.append(f"NCA: no data for DV '{dv}' — skipped.")
            continue

        # Align lengths and remove NaN pairs
        min_len = min(len(x_arr), len(y_arr))
        x_arr, y_arr = x_arr[:min_len], y_arr[:min_len]
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_c, y_c = x_arr[mask], y_arr[mask]
        n_obs = int(mask.sum())

        if n_obs < 10:
            warnings.append(
                f"NCA: only {n_obs} valid observations for {iv}→{dv} — skipped."
            )
            continue

        # ── CE-FDH ──────────────────────────────────────────────────────────
        try:
            ce_d, ceil_x_raw, ceil_y_raw = _ce_fdh(x_c, y_c)
        except Exception as exc:
            warnings.append(f"NCA CE-FDH failed for {iv}→{dv}: {exc}")
            ce_d, ceil_x_raw, ceil_y_raw = 0.0, [], []

        # ── CR-FDH ──────────────────────────────────────────────────────────
        try:
            cr_d, cr_slope, cr_intercept, _, _ = _cr_fdh(x_c, y_c)
        except Exception as exc:
            warnings.append(f"NCA CR-FDH failed for {iv}→{dv}: {exc}")
            cr_d, cr_slope, cr_intercept = 0.0, 0.0, 0.0

        # ── Permutation test ─────────────────────────────────────────────────
        perm_ce: list[float] = []
        perm_cr: list[float] = []

        _emit(log_fn, "info",
              f"    Permutation test ({n_permutations} samples)")

        for _ in range(n_permutations):
            y_perm = rng.permutation(y_c)
            try:
                d_ce_p, *_ = _ce_fdh(x_c, y_perm)
                perm_ce.append(d_ce_p)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                pass
            try:
                d_cr_p, *_ = _cr_fdh(x_c, y_perm)
                perm_cr.append(d_cr_p)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                pass

        ce_p = float(np.mean(np.array(perm_ce) >= ce_d)) if perm_ce else None
        cr_p = float(np.mean(np.array(perm_cr) >= cr_d)) if perm_cr else None

        # Significant if either ceiling p < 0.05
        significant = (
            (ce_p is not None and ce_p < 0.05)
            or (cr_p is not None and cr_p < 0.05)
        )

        # ── Sample scatter points for frontend (max 200 pts) ────────────────
        if n_obs > 200:
            idx_sample = rng.choice(n_obs, size=200, replace=False)
            sc_x = [round(float(x_c[i]), 4) for i in sorted(idx_sample)]
            sc_y = [round(float(y_c[i]), 4) for i in sorted(idx_sample)]
        else:
            sc_x = [round(float(v), 4) for v in x_c]
            sc_y = [round(float(v), 4) for v in y_c]

        # Sample ceiling line points (max 100 pts)
        ceil_x_out = [round(v, 4) for v in ceil_x_raw[:100]]
        ceil_y_out = [round(v, 4) for v in ceil_y_raw[:100]]

        entries.append(NCAEntry(
            iv=iv, dv=dv, n_obs=n_obs,
            ce_fdh_d=ce_d,
            ce_fdh_label=_nca_label(ce_d),
            ce_fdh_p=round(ce_p, 4) if ce_p is not None else None,
            cr_fdh_d=cr_d,
            cr_fdh_label=_nca_label(cr_d),
            cr_fdh_slope=cr_slope,
            cr_fdh_intercept=cr_intercept,
            cr_fdh_p=round(cr_p, 4) if cr_p is not None else None,
            significant=significant,
            scatter_x=sc_x,
            scatter_y=sc_y,
            ceiling_x=ceil_x_out,
            ceiling_y=ceil_y_out,
        ))

        _emit(log_fn, "ok",
              f"    {iv}→{dv}: CE-FDH d={ce_d:.4f} ({_nca_label(ce_d)})  "
              f"CR-FDH d={cr_d:.4f} ({_nca_label(cr_d)})  "
              f"sig={significant}")

    if not entries:
        raise ValueError(
            "NCA: no IV→DV pairs could be computed. "
            "Check that the model has structural paths and the "
            "dataset contains all indicator columns."
        )

    _emit(log_fn, "ok",
          f"NCA complete — {len(entries)} pair(s) analysed")

    return NCAResult(
        entries=entries,
        n_permutations=n_permutations,
        warnings=warnings,
    )
