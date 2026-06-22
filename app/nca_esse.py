"""
nca_esse.py  —  NAVAL-SEM v0.9
=================================
Necessary Condition Analysis — Effect Size Sensitivity Extension (NCA-ESSE).

Becker, J.-M., Richter, N. F., Ringle, C. M., & Sarstedt, M. (2026).
Must-have, or maybe not? A sensitivity-based extension to necessary
condition analysis. Journal of Business Research, 206, 115920.
https://doi.org/10.1016/j.jbusres.2025.115920  (CC BY 4.0)

Concept
-------
Standard (deterministic) NCA falsifies a necessary condition the moment a
single observation lands in the "ceiling zone" — the empty upper-left
region of the X–Y scatter that signifies necessity. With large samples,
a tiny share of atypical response combinations can mask a true
necessary condition entirely (effect size collapses to 0).

NCA-ESSE replaces the all-or-nothing deterministic ceiling with a
*typicality* perspective: it sweeps an ECDF threshold p (e.g. 0–5%) and,
at each step, recomputes the ceiling line after discarding the most
extreme p% of the joint empirical distribution's "violating" mass
(low-X / high-Y combinations). The resulting empirical sensitivity curve
is compared against the same sweep run on a joint *uniform* benchmark
distribution (where no necessity relationship exists by construction) to
distinguish genuine necessity from a random artefact of shifting the
ceiling line.

This module is independent of nca.py's public API surface (it reuses the
private `_ce_fdh` / `_cr_fdh` ceiling-line primitives) and returns its own
`NCAESSEResult`, mirroring the fimix.py / plspos.py composition pattern
used elsewhere in this codebase.

Public API
----------
  compute_nca_esse(df, model_syntax, thresholds, n_permutations,
                    n_benchmark_reps, seed, log_fn) -> NCAESSEResult
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _emit, _safe_float
from app.nca import _ce_fdh, _cr_fdh, _nca_label
from app.parser import parse_lavaan
from app.schemas import NCAESSEEntry, NCAESSEResult, NCAESSEThresholdPoint

logger = logging.getLogger("naval_sem.nca_esse")

DEFAULT_THRESHOLDS = [round(0.005 * i, 4) for i in range(0, 11)]  # 0.0 .. 0.05


# ── Core threshold-removal sweep ─────────────────────────────────────────────

def _threshold_sweep(
    x: np.ndarray,
    y: np.ndarray,
    thresholds: list,
) -> dict:
    """
    Single iterative-removal pass producing CE-FDH and CR-FDH effect sizes
    at each cumulative ECDF threshold checkpoint.

    At each step, the most "extreme" remaining ceiling-defining point —
    the lowest-X / highest-Y peak cell, i.e. the joint ECDF's worst
    necessity violator — is removed in full (all tied observations at
    that exact (x, y) coordinate, since removing only some would not
    move a staircase ceiling built on discrete/Likert data). The ceiling
    is recomputed and the running removed-fraction checked against each
    requested threshold in ascending order.

    Returns
    -------
    {threshold: {"ce_d": float, "cr_d": float, "pct_excluded": float}}
    """
    n = len(x)
    x_w, y_w = x.copy(), y.copy()
    sorted_thresh = sorted(thresholds)
    out: dict = {}
    next_idx = 0
    removed = 0

    def _snapshot(pct_excluded: float):
        ce_d, *_ = _ce_fdh(x_w, y_w)
        try:
            cr_d, *_ = _cr_fdh(x_w, y_w)
        except Exception:
            cr_d = 0.0
        return {"ce_d": ce_d, "cr_d": cr_d, "pct_excluded": pct_excluded}

    # threshold 0.0 == standard deterministic CE-FDH / CR-FDH
    while next_idx < len(sorted_thresh) and sorted_thresh[next_idx] <= 1e-12:
        out[sorted_thresh[next_idx]] = _snapshot(0.0)
        next_idx += 1

    while next_idx < len(sorted_thresh):
        if len(x_w) < 5:
            last = out[sorted_thresh[next_idx - 1]] if next_idx > 0 else _snapshot(removed / n)
            while next_idx < len(sorted_thresh):
                out[sorted_thresh[next_idx]] = last
                next_idx += 1
            break

        _, _, _, peaks = _ce_fdh(x_w, y_w)
        if not peaks:
            snap = _snapshot(removed / n)
            while next_idx < len(sorted_thresh):
                out[sorted_thresh[next_idx]] = snap
                next_idx += 1
            break

        worst_x, worst_y = peaks[0]   # smallest x among current peaks = worst violator
        mask = (x_w == worst_x) & (y_w == worst_y)
        cell_size = int(mask.sum())
        target = sorted_thresh[next_idx] * n

        if removed + cell_size > target and removed > 0:
            out[sorted_thresh[next_idx]] = _snapshot(removed / n)
            next_idx += 1
            continue

        x_w, y_w = x_w[~mask], y_w[~mask]
        removed += cell_size

        while next_idx < len(sorted_thresh) and removed >= sorted_thresh[next_idx] * n - 1e-9:
            out[sorted_thresh[next_idx]] = _snapshot(removed / n)
            next_idx += 1

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def compute_nca_esse(
    df: pd.DataFrame,
    model_syntax: str,
    thresholds: Optional[list] = None,
    n_permutations: int = 200,
    n_benchmark_reps: int = 200,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> NCAESSEResult:
    """
    NCA with Effect Size Sensitivity Extension (NCA-ESSE) across all
    structural IV → DV pairs.

    For each pair X → Y:

    1. Sweep ECDF thresholds (default 0–5 %, 0.5 pt increments), removing
       the most extreme violating cells at each step and recomputing the
       CE-FDH / CR-FDH ceiling lines and effect sizes.
    2. Run the same sweep on ``n_benchmark_reps`` joint-uniform random
       samples (same n, same scope) as a theoretical no-necessity
       benchmark.
    3. Permutation-test (shuffled Y) the empirical effect size at every
       threshold.
    4. Recommend a threshold: the largest one reachable through a
       contiguous run (from the first nonzero threshold) where the
       empirical effect-size gain exceeds the benchmark's gain.

    Parameters
    ----------
    df               : pd.DataFrame
    model_syntax     : str    lavaan syntax.
    thresholds       : list[float] | None   Defaults to 0–5% in 0.5pt steps.
    n_permutations   : int    Permutation samples per threshold (per pair).
    n_benchmark_reps : int    Joint-uniform benchmark replications (per pair).
    seed             : int
    log_fn           : callable | None

    Returns
    -------
    NCAESSEResult
    """
    thresholds = sorted(thresholds) if thresholds else DEFAULT_THRESHOLDS
    _emit(log_fn, "step",
          f"NCA-ESSE: parsing model ({len(thresholds)} thresholds, "
          f"{n_permutations} permutations, {n_benchmark_reps} benchmark reps)")

    parsed      = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    structural  = parsed.get("structural", [])
    warnings:   list[str] = []

    if not structural:
        raise ValueError("NCA-ESSE requires at least one structural path.")

    rng = np.random.default_rng(seed)
    composites = _build_composites(df, measurement, structural)

    def _get_scores(name: str) -> Optional[np.ndarray]:
        if name in composites:
            return composites[name].values
        if name in df.columns:
            return df[name].astype(float).values
        return None

    entries: list[NCAESSEEntry] = []
    seen_pairs: set = set()

    for rel in structural:
        iv, dv = rel["rhs"], rel["lhs"]
        if (iv, dv) in seen_pairs:
            continue
        seen_pairs.add((iv, dv))

        _emit(log_fn, "info", f"  NCA-ESSE: {iv} → {dv}")
        entry_warnings: list[str] = []

        x_arr, y_arr = _get_scores(iv), _get_scores(dv)
        if x_arr is None or y_arr is None:
            warnings.append(f"NCA-ESSE: missing data for {iv}→{dv} — skipped.")
            continue

        min_len = min(len(x_arr), len(y_arr))
        x_arr, y_arr = x_arr[:min_len], y_arr[:min_len]
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_c, y_c = x_arr[mask], y_arr[mask]
        n_obs = int(mask.sum())

        if n_obs < 30:
            warnings.append(
                f"NCA-ESSE: only {n_obs} valid observations for {iv}→{dv} "
                "(need ≥30 for a meaningful threshold sweep) — skipped."
            )
            continue

        # ── Empirical sweep ─────────────────────────────────────────────────
        try:
            emp = _threshold_sweep(x_c, y_c, thresholds)
        except Exception as exc:
            warnings.append(f"NCA-ESSE empirical sweep failed for {iv}→{dv}: {exc}")
            continue

        # ── Benchmark sweep (joint uniform, same scope & n) ───────────────────
        x_min, x_max = float(x_c.min()), float(x_c.max())
        y_min, y_max = float(y_c.min()), float(y_c.max())
        bench_acc = {t: [] for t in thresholds}
        for _ in range(n_benchmark_reps):
            try:
                xb = rng.uniform(x_min, x_max, size=n_obs)
                yb = rng.uniform(y_min, y_max, size=n_obs)
                sweep_b = _threshold_sweep(xb, yb, thresholds)
                for t in thresholds:
                    bench_acc[t].append(sweep_b[t]["ce_d"])
            except Exception:
                continue
        theo = {
            t: float(np.mean(vals)) if vals else 0.0
            for t, vals in bench_acc.items()
        }

        # ── Permutation test (shuffled Y) at every threshold ──────────────────
        perm_acc = {t: [] for t in thresholds}
        for _ in range(n_permutations):
            try:
                y_perm = rng.permutation(y_c)
                sweep_p = _threshold_sweep(x_c, y_perm, thresholds)
                for t in thresholds:
                    perm_acc[t].append(sweep_p[t]["ce_d"])
            except Exception:
                continue

        # ── Assemble per-threshold points + deltas ───────────────────────────
        points: list[NCAESSEThresholdPoint] = []
        prev_emp, prev_theo = None, None
        for t in thresholds:
            emp_d = round(float(emp[t]["ce_d"]), 6)
            theo_d = round(float(theo.get(t, 0.0)), 6)
            pct_excl = round(float(emp[t]["pct_excluded"]), 6)

            perm_vals = perm_acc.get(t, [])
            p_val = (
                round(float(np.mean(np.array(perm_vals) >= emp_d)), 4)
                if perm_vals else None
            )
            sig = p_val is not None and p_val < 0.05

            d_emp = round(emp_d - prev_emp, 6) if prev_emp is not None else None
            d_theo = round(theo_d - prev_theo, 6) if prev_theo is not None else None
            d_diff = round(d_emp - d_theo, 6) if (d_emp is not None and d_theo is not None) else None

            points.append(NCAESSEThresholdPoint(
                threshold=t,
                pct_excluded=pct_excl,
                empirical_d=emp_d,
                theoretical_d=theo_d,
                delta_empirical=d_emp,
                delta_theoretical=d_theo,
                delta_diff=d_diff,
                p_value=p_val,
                significant=sig,
            ))
            prev_emp, prev_theo = emp_d, theo_d

        # ── Recommended threshold: longest contiguous run (from the first
        #    nonzero step) where the empirical gain beats the benchmark gain ──
        recommended = points[0]
        for pt in points[1:]:
            if pt.delta_diff is not None and pt.delta_diff > 0:
                recommended = pt
            else:
                break

        if recommended.threshold == 0.0:
            entry_warnings.append(
                f"NCA-ESSE: no threshold above 0% showed a benchmark-beating "
                f"gain for {iv}→{dv}; standard NCA result stands."
            )

        # ── Ceiling line at the recommended threshold (for plotting) ─────────
        ceil_x, ceil_y = [], []
        try:
            x_trim, y_trim = x_c.copy(), y_c.copy()
            n_remove = int(round(recommended.pct_excluded * n_obs))
            removed_n = 0
            while removed_n < n_remove and len(x_trim) >= 5:
                _, _, _, pk = _ce_fdh(x_trim, y_trim)
                if not pk:
                    break
                wx, wy = pk[0]
                m = (x_trim == wx) & (y_trim == wy)
                x_trim, y_trim = x_trim[~m], y_trim[~m]
                removed_n += int(m.sum())
            _, px, py, _ = _ce_fdh(x_trim, y_trim)
            ceil_x = [round(v, 4) for v in px[:100]]
            ceil_y = [round(v, 4) for v in py[:100]]
        except Exception as exc:
            entry_warnings.append(f"NCA-ESSE: ceiling line reconstruction failed: {exc}")

        entries.append(NCAESSEEntry(
            iv=iv, dv=dv, n_obs=n_obs,
            thresholds=points,
            recommended_threshold=recommended.threshold,
            recommended_effect_size=recommended.empirical_d,
            recommended_label=_nca_label(recommended.empirical_d),
            ceiling_x=ceil_x,
            ceiling_y=ceil_y,
            warnings=entry_warnings,
        ))

        _emit(log_fn, "ok",
              f"    {iv}→{dv}: d(0%)={points[0].empirical_d:.4f} → "
              f"d({recommended.threshold:.1%})={recommended.empirical_d:.4f} "
              f"({_nca_label(recommended.empirical_d)})")

    if not entries:
        raise ValueError(
            "NCA-ESSE: no IV→DV pairs could be computed. "
            "Check that the model has structural paths and the dataset "
            "contains all indicator columns, with ≥30 valid observations "
            "per pair."
        )

    _emit(log_fn, "ok", f"NCA-ESSE complete — {len(entries)} pair(s) analysed")

    return NCAESSEResult(
        entries=entries,
        threshold_range=thresholds,
        benchmark="joint_uniform",
        n_permutations=n_permutations,
        n_benchmark_reps=n_benchmark_reps,
        warnings=warnings,
    )
