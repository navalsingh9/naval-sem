"""
engine_ipma.py  —  NAVAL-SEM v1.1
===================================
Importance-Performance Map Analysis (IPMA).

Public API
----------
  compute_ipma(df, model_syntax, target_lv, algorithm,
               scale_min, scale_max, log_fn) -> IPMAResult

Internal
--------
  _generate_ipma_chart(indicator_entries, target_lv) -> (chart_svg, chart_png)
      Indicator-level quadrant map (v1.1 / B2). Called from compute_ipma;
      populates IPMAResult.chart_svg / .chart_png. Requires matplotlib.

Reference
---------
  Ringle & Sarstedt (2016); Hair et al. (2022, Chapter 7);
  Martilla & James (1977) for the original IPA quadrant framework.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot is imported (B2)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _ci_from_bootstrap, _coef_from_params, _emit, _safe_float, _sig_from_ci
from app.engine import compute_indirect_effects, fit_model
from app.parser import parse_lavaan
from app.schemas import IPMAEntry, IPMAIndicatorEntry, IPMAResult, PathParameter

logger = logging.getLogger("naval_sem.ipma")


# ── chart generation (B2) ────────────────────────────────────────────────────

# Standard IPMA quadrant names (Martilla & James, 1977; adapted for PLS-SEM
# per Hair et al., 2022, Ch. 7). Keys describe the (performance, importance)
# corner each label sits in, relative to the mean-importance / mean-performance
# divider lines.
_QUADRANT_LABELS = {
    "top_right":    "Keep up the good work",
    "top_left":     "Concentrate here",
    "bottom_right": "Possible overkill",
    "bottom_left":  "Low priority",
}


def _generate_ipma_chart(
    indicator_entries: list[IPMAIndicatorEntry],
    target_lv: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Render the indicator-level IPMA quadrant map: each indicator plotted at
    (performance, importance), with dashed divider lines at the mean of
    each axis (computed over the plotted indicators) splitting the chart
    into the four standard IPMA quadrants.

    Parameters
    ----------
    indicator_entries : list[IPMAIndicatorEntry]
    target_lv          : str   Used in the chart title only.

    Returns
    -------
    (chart_svg, chart_png) : tuple[str | None, str | None]
        chart_svg — inline SVG markup (XML prolog stripped), for the web
                    frontend to drop straight into the DOM.
        chart_png — base64-encoded PNG bytes (150 dpi), for DOCX/PDF
                    embedding — see export_docx / export_pdf.
        Both are None when there are no plottable (non-null) points.
    """
    points = [
        (e.indicator, _safe_float(e.performance), _safe_float(e.importance))
        for e in indicator_entries
    ]
    points = [(ind, x, y) for ind, x, y in points if x is not None and y is not None]
    if not points:
        return None, None

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    ax.scatter(xs, ys, s=48, color="#2A6F97", zorder=3,
               edgecolors="white", linewidths=0.6)
    for ind, x, y in points:
        ax.annotate(ind, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#1B1B1B")

    # Pad the axes so points, labels, and the mean lines all stay visible —
    # including degenerate cases (a single indicator, or several sharing the
    # same performance/importance), where span would otherwise be zero.
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(x_span * 0.15, 3.0)
    y_pad = max(y_span * 0.15, 0.02)
    x_lo, x_hi = min(min(xs), mean_x) - x_pad, max(max(xs), mean_x) + x_pad
    y_lo, y_hi = min(min(ys), mean_y) - y_pad, max(max(ys), mean_y) + y_pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.axvline(mean_x, color="#888888", linestyle="--", linewidth=1, zorder=1)
    ax.axhline(mean_y, color="#888888", linestyle="--", linewidth=1, zorder=1)

    qs = dict(fontsize=9, color="#666666", style="italic")
    pf = 0.03  # inset quadrant labels 3% from each edge so they never clip
    ax.text(x_hi - (x_hi - x_lo) * pf, y_hi - (y_hi - y_lo) * pf,
            _QUADRANT_LABELS["top_right"],    ha="right", va="top",    **qs)
    ax.text(x_lo + (x_hi - x_lo) * pf, y_hi - (y_hi - y_lo) * pf,
            _QUADRANT_LABELS["top_left"],     ha="left",  va="top",    **qs)
    ax.text(x_hi - (x_hi - x_lo) * pf, y_lo + (y_hi - y_lo) * pf,
            _QUADRANT_LABELS["bottom_right"], ha="right", va="bottom", **qs)
    ax.text(x_lo + (x_hi - x_lo) * pf, y_lo + (y_hi - y_lo) * pf,
            _QUADRANT_LABELS["bottom_left"],  ha="left",  va="bottom", **qs)

    ax.set_xlabel("Performance")
    ax.set_ylabel("Importance (total effect)")
    ax.set_title(f"Indicator-Level IPMA — Target: {target_lv}")
    fig.tight_layout()

    # SVG — primary, for the web frontend
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg")
    chart_svg = svg_buf.getvalue()
    svg_start = chart_svg.find("<svg")
    if svg_start > 0:
        chart_svg = chart_svg[svg_start:]   # drop the XML prolog/doctype

    # PNG — for DOCX/PDF embedding
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    chart_png = base64.b64encode(png_buf.getvalue()).decode("ascii")

    plt.close(fig)  # release the figure — long-running server process
    return chart_svg, chart_png


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

    _coef_map = {(p.rhs, p.lhs): p.estimate for p in res.parameters if p.op == "~"}

    try:
        # existing_coef_map is passed so compute_indirect_effects skips its internal
        # fit_model call entirely (it uses the map directly for path products).
        # n_bootstrap=0 additionally skips the bootstrap loop.
        # Do NOT remove either argument — doing so would cause a second full fit.
        indirect_res  = compute_indirect_effects(
            df, model_syntax,
            n_bootstrap=0, log_fn=None,
            algorithm=algorithm,
            existing_coef_map=_coef_map,
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

    # ── Indicator-level IPMA (Hair et al. 2022, Ch. 7) ────────────────────────
    _indicator_entries = []
    _loading_map: dict[tuple, float] = {}
    for _p in res.parameters:
        if getattr(_p, "op", None) == "=~":
            _loading_map[(_p.lhs, _p.rhs)] = abs(_p.estimate)

    for entry in entries:
        _lv = entry.lv
        _te = entry.importance   # total effect on target
        _inds = measurement.get(_lv, [])
        for _ind in _inds:
            _lam = _loading_map.get((_lv, _ind), 0.0)
            _ind_importance = round(float(_te * _lam), 6)
            if _ind in df.columns:
                _raw_mean = float(df[_ind].mean())
            else:
                _raw_mean = (eff_min + eff_max) / 2
            _ind_perf = max(0.0, min(100.0,
                round((_raw_mean - eff_min) / scale_range * 100, 2)))
            _indicator_entries.append(IPMAIndicatorEntry(
                lv=_lv, indicator=_ind,
                importance=_ind_importance,
                performance=_ind_perf,
            ))

    # ── Indicator-level IPMA quadrant chart (B2) ──────────────────────────────
    _emit(log_fn, "step", "IPMA: generating quadrant chart")
    try:
        _chart_svg, _chart_png = _generate_ipma_chart(_indicator_entries, target_lv)
        if _chart_svg is None:
            warnings.append(
                "IPMA: no indicator-level data to chart; chart_svg/chart_png "
                "are empty."
            )
    except Exception as exc:
        warnings.append(f"IPMA: chart generation failed ({exc}). "
                         "chart_svg/chart_png are empty.")
        _chart_svg, _chart_png = None, None

    _emit(log_fn, "ok",
          f"IPMA complete — {len(entries)} predictors of '{target_lv}'")

    return IPMAResult(
        target_lv=target_lv,
        entries=entries,
        scale_min=eff_min,
        scale_max=eff_max,
        algorithm=algorithm,
        warnings=warnings,
        indicator_entries=_indicator_entries,
        chart_svg=_chart_svg,
        chart_png=_chart_png,
    )
