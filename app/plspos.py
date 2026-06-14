"""
plspos.py  —  NAVAL-SEM v0.8
==============================
PLS-POS: Prediction-Oriented Segmentation.

Reference: Becker et al. (2013). Discovering Unobserved Heterogeneity
in Structural Equation Models to Avert Validity Threats.
MIS Quarterly, 37(3), 665–694.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _emit, _safe_float
from app.fimix import FIMIXResult, _ols_paths
from app.parser import parse_lavaan
from app.schemas import PLSPOSResult, PLSPOSSegment

logger = logging.getLogger("naval_sem.plspos")


# ── private helpers ───────────────────────────────────────────────────────────

def _total_within_r2(
    hard_assign: np.ndarray,
    eq_data: list,
    K: int,
) -> float:
    """Sum of within-segment R² across all equations and segments (objective to maximise)."""
    total = 0.0
    for k in range(K):
        mask = hard_assign == k
        if mask.sum() < 2:
            continue
        for eq in eq_data:
            y = eq["y"][mask]
            X = eq["X"][mask]
            coefs, r2, _ = _ols_paths(X, y, np.ones(len(y)))
            total += r2
    return total


def _pos_iterate(
    composites: dict,
    struct_rels: list,
    n: int,
    K: int,
    init_assign: np.ndarray,
    max_iter: int = 200,
) -> tuple:
    """
    PLS-POS iteration from an initial hard assignment.

    Returns
    -------
    (hard_assign, eq_data, seg_coefs)
    """
    # Build predictor/outcome matrices per structural equation
    eq_data = []
    for rel in struct_rels:
        lhs = rel["lhs"]
        rhs_list = rel["rhs"] if isinstance(rel["rhs"], list) else [rel["rhs"]]
        y_col = composites.get(lhs)
        X_cols = [composites.get(r) for r in rhs_list if composites.get(r) is not None]
        if y_col is None or not X_cols:
            continue
        y = y_col.values.astype(float)
        X = np.column_stack([c.values.astype(float) for c in X_cols])
        eq_data.append({"lhs": lhs, "rhs": rhs_list[: len(X_cols)], "y": y, "X": X})

    if not eq_data:
        raise ValueError("PLS-POS: no complete structural equations found.")

    hard = init_assign.copy()
    prev_obj = -np.inf

    for _iter in range(max_iter):
        # ── Estimate segment models ──────────────────────────────────────────
        seg_coefs: list[list] = []
        for k in range(K):
            mask = hard == k
            kc = []
            for eq in eq_data:
                w = mask.astype(float)
                if w.sum() < 2:
                    kc.append(np.zeros(eq["X"].shape[1]))
                else:
                    c, _, _ = _ols_paths(eq["X"], eq["y"], w)
                    kc.append(c)
            seg_coefs.append(kc)

        # ── Reassign observations to segment minimising prediction SSE ───────
        new_hard = np.zeros(n, dtype=int)
        for i in range(n):
            best_k, best_sse = 0, np.inf
            for k in range(K):
                sse = 0.0
                for eq_i, eq in enumerate(eq_data):
                    pred = eq["X"][i] @ seg_coefs[k][eq_i]
                    sse += (eq["y"][i] - pred) ** 2
                if sse < best_sse:
                    best_sse, best_k = sse, k
            new_hard[i] = best_k

        obj = _total_within_r2(new_hard, eq_data, K)

        if np.array_equal(new_hard, hard) or abs(obj - prev_obj) < 1e-8:
            hard = new_hard
            break

        hard, prev_obj = new_hard, obj

    return hard, eq_data, seg_coefs


# ── public API ────────────────────────────────────────────────────────────────

def run_plspos(
    df: pd.DataFrame,
    model_syntax: str,
    k: int,
    fimix_result: Optional[FIMIXResult] = None,
    n_starts: int = 10,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> PLSPOSResult:
    """
    Run PLS-POS for a fixed number of segments K.

    Parameters
    ----------
    df            : Tidy observation-level DataFrame.
    model_syntax  : lavaan-style model string.
    k             : Number of segments (fixed — use FIMIX to select K first).
    fimix_result  : Optional FIMIXResult; used to warm-start the first run.
    n_starts      : Number of random restarts (first start uses FIMIX init if available).
    seed          : Master RNG seed for reproducibility.
    log_fn        : Optional callback(level, msg) for streaming logs.

    Returns
    -------
    PLSPOSResult with per-segment path coefficients, R², and stability scores.
    """
    if k < 2:
        raise ValueError(
            f"PLS-POS requires k ≥ 2 segments (got k={k}). "
            "Use FIMIX to determine an appropriate number of segments."
        )

    _emit(log_fn, "step", f"PLS-POS: K={k}, {n_starts} starts")

    parsed = parse_lavaan(model_syntax)
    composites = _build_composites(
        df, parsed.get("measurement", {}), parsed.get("structural", [])
    )
    struct_rels = parsed.get("structural", [])
    n = len(df)
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    best_obj: float = -np.inf
    best_assign: Optional[np.ndarray] = None
    best_eq_data: Optional[list] = None
    best_coefs: Optional[list] = None
    all_assignments: list[np.ndarray] = []

    for s in range(n_starts):
        # Warm-start from FIMIX hard assignment on the first run when available
        if s == 0 and fimix_result is not None:
            sol = next((x for x in fimix_result.solutions if x.k == k), None)
            if sol is not None:
                sizes = [seg.size for seg in sol.segments]
                init = np.repeat(np.arange(k), sizes)[:n]
                np.random.default_rng(seed).shuffle(init)
            else:
                init = rng.integers(0, k, size=n)
        else:
            init = rng.integers(0, k, size=n)

        try:
            assign, eq_data, seg_coefs = _pos_iterate(
                composites, struct_rels, n, k, init
            )
        except Exception as exc:
            warnings.append(f"PLS-POS start {s} failed: {exc}")
            continue

        obj = _total_within_r2(assign, eq_data, k)
        all_assignments.append(assign.copy())

        if obj > best_obj:
            best_obj = obj
            best_assign = assign.copy()
            best_eq_data = eq_data
            best_coefs = seg_coefs

    if best_assign is None:
        raise ValueError("PLS-POS: all starts failed.")

    # ── Stability: proportion of starts that agree with best (per observation) ─
    if len(all_assignments) > 1:
        agreement = float(
            np.mean([
                np.mean(a == best_assign)
                for a in all_assignments
            ])
        )
    else:
        agreement = 0.0   # single run — no cross-run evidence of stability

    # ── Build PLSPOSSegment objects ───────────────────────────────────────────
    segs: list[PLSPOSSegment] = []
    for k_i in range(k):
        mask = best_assign == k_i
        paths: dict[str, float] = {}
        r2s: dict[str, float] = {}

        for eq_i, eq in enumerate(best_eq_data):
            for rhs_name, c in zip(eq["rhs"], best_coefs[k_i][eq_i]):
                paths[f"{eq['lhs']}~{rhs_name}"] = round(float(c), 6)

            y_seg = eq["y"][mask]
            x_seg = eq["X"][mask]
            if len(y_seg) > 1:
                y_hat = x_seg @ best_coefs[k_i][eq_i]
                ss_res = float(np.sum((y_seg - y_hat) ** 2))
                ss_tot = float(np.sum((y_seg - y_seg.mean()) ** 2))
                r2s[eq["lhs"]] = round(
                    float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-14))), 6
                )

        segs.append(
            PLSPOSSegment(
                segment_id=k_i + 1,
                size=int(mask.sum()),
                path_coefficients=paths,
                r_squared=r2s,
                stability=round(float(agreement), 4),
            )
        )

    # ── FIMIX comparison table ────────────────────────────────────────────────
    fimix_cmp: Optional[Dict[str, Any]] = None
    if fimix_result is not None:
        sol = next((x for x in fimix_result.solutions if x.k == k), None)
        if sol is not None:
            fimix_cmp = {
                "fimix_paths": {
                    f"seg{seg.segment_id}": seg.path_coefficients
                    for seg in sol.segments
                },
                "pos_paths": {
                    f"seg{seg.segment_id}": seg.path_coefficients
                    for seg in segs
                },
            }

    _emit(log_fn, "ok", f"PLS-POS complete — K={k}, stability={agreement:.2f}")

    return PLSPOSResult(
        k=k,
        segments=segs,
        fimix_comparison=fimix_cmp,
        n_obs=n,
        warnings=warnings,
    )
