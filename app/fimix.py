"""
fimix.py  —  NAVAL-SEM v0.8
============================
FIMIX-PLS: Finite Mixture PLS segmentation.

Reference: Hahn et al. (2002); Sarstedt et al. (2011).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _build_composites, _emit, _safe_float
from app.parser import parse_lavaan
from app.schemas import FIMIXResult, FIMIXSolution, FIMIXSegment

logger = logging.getLogger("naval_sem.fimix")


# ── private helpers ──────────────────────────────────────────────────────────

def _ols_paths(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
    """Weighted OLS: returns (coefs, r2, sigma2). X has no intercept column."""
    W = np.diag(weights)
    XtW = X.T @ W
    try:
        coefs = np.linalg.solve(XtW @ X, XtW @ y)
    except np.linalg.LinAlgError:
        coefs = np.zeros(X.shape[1])
    y_hat  = X @ coefs
    ss_res = float(np.sum(weights * (y - y_hat) ** 2))
    ss_tot = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    r2     = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-14))
    sigma2 = max(ss_res / max(np.sum(weights), 1e-12), 1e-14)
    return coefs, r2, sigma2


def _relative_entropy(tau: np.ndarray) -> float:
    """R_E in [0,1]. tau: (n, K) posterior membership matrix."""
    n, K = tau.shape
    if K < 2 or n == 0:
        return 1.0
    H = -np.sum(tau * np.log(np.clip(tau, 1e-300, 1.0))) / (n * np.log(K))
    return round(float(1.0 - H), 6)


def _run_em(composites, structural_rels, n, K, rng, max_iter, tol):
    """
    Single EM run for FIMIX-PLS.

    Returns
    -------
    (tau, seg_params, eq_data, pi_k, log_lik)
    Raises ValueError when no complete structural equations can be built.
    """
    # Build predictor/outcome matrices per structural equation
    eq_data = []
    for rel in structural_rels:
        lhs      = rel["lhs"]
        rhs_list = rel["rhs"] if isinstance(rel["rhs"], list) else [rel["rhs"]]
        y_col    = composites.get(lhs)
        X_cols   = [composites.get(r) for r in rhs_list if composites.get(r) is not None]
        if y_col is None or not X_cols:
            continue
        y = y_col.values.astype(float)
        X = np.column_stack([c.values.astype(float) for c in X_cols])
        eq_data.append({"lhs": lhs, "rhs": rhs_list[: len(X_cols)], "y": y, "X": X})

    if not eq_data:
        raise ValueError("FIMIX: no complete structural equations found.")

    # Random initialisation of posterior memberships
    raw  = rng.dirichlet(np.ones(K), size=n)
    tau  = raw / raw.sum(axis=1, keepdims=True)
    pi_k = tau.mean(axis=0)

    # Per-segment regression parameters: list of K dicts, each holding
    # one entry per equation with keys "coefs" and "sigma2".
    seg_params = [{} for _ in range(K)]

    log_lik = -np.inf

    for _iter in range(max_iter):

        # ── M-step ───────────────────────────────────────────────────────────
        pi_k = tau.mean(axis=0)
        for k in range(K):
            w = tau[:, k]
            if w.sum() < 1e-6:
                # Segment has collapsed — reinitialise with uniform weights to escape degeneracy
                seg_params[k] = [{"coefs": np.zeros(eq["X"].shape[1]), "sigma2": 1.0} for eq in eq_data]
                continue
            seg_params[k] = []
            for eq in eq_data:
                coefs, _, sigma2 = _ols_paths(eq["X"], eq["y"], w)
                seg_params[k].append({"coefs": coefs, "sigma2": sigma2})

        # ── E-step ───────────────────────────────────────────────────────────
        log_p = np.zeros((n, K))
        for k in range(K):
            lp = np.log(max(pi_k[k], 1e-300))
            for eq_i, eq in enumerate(eq_data):
                c   = seg_params[k][eq_i]["coefs"]
                s2  = seg_params[k][eq_i]["sigma2"]
                resid = eq["y"] - eq["X"] @ c
                lp    = lp - 0.5 * np.log(2 * np.pi * s2) - 0.5 * resid ** 2 / s2
            log_p[:, k] = lp

        log_p_max = log_p.max(axis=1, keepdims=True)
        p_stable  = np.exp(log_p - log_p_max)
        row_sums  = p_stable.sum(axis=1, keepdims=True)
        tau_new   = p_stable / np.maximum(row_sums, 1e-300)
        new_ll    = float(
            np.sum(log_p_max.ravel() + np.log(np.maximum(row_sums.ravel(), 1e-300)))
        )

        delta   = abs(new_ll - log_lik)
        tau     = tau_new
        log_lik = new_ll

        if delta < tol:
            break

    return tau, seg_params, eq_data, pi_k, log_lik


# ── public API ───────────────────────────────────────────────────────────────

def run_fimix(
    df: pd.DataFrame,
    model_syntax: str,
    k_max: int = 5,
    n_starts: int = 10,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> FIMIXResult:
    """
    Run FIMIX-PLS segmentation for K = 2 … k_max.

    Parameters
    ----------
    df            : Tidy observation-level DataFrame.
    model_syntax  : lavaan-style model string.
    k_max         : Maximum number of segments to test (inclusive).
    n_starts      : Random restarts per K to escape local optima.
    max_iter      : EM iteration cap per run.
    tol           : Log-likelihood convergence tolerance.
    seed          : Master RNG seed (reproducibility).
    log_fn        : Optional callback(level, msg) for streaming logs.

    Returns
    -------
    FIMIXResult with solutions for each converged K and the recommended K
    chosen by minimum CAIC.
    """
    _emit(log_fn, "step", f"FIMIX-PLS: K=2..{k_max}, {n_starts} starts, {max_iter} iter")

    parsed      = parse_lavaan(model_syntax)
    composites  = _build_composites(df, parsed.get("measurement", {}), parsed.get("structural", []))
    struct_rels = parsed.get("structural", [])
    n           = len(df)
    rng         = np.random.default_rng(seed)
    warnings: list[str] = []
    solutions: list[FIMIXSolution] = []

    # Number of free structural-path parameters per segment
    n_params_base = sum(
        len(r["rhs"]) if isinstance(r["rhs"], list) else 1
        for r in struct_rels
    )

    for K in range(2, k_max + 1):
        _emit(log_fn, "step", f"  FIMIX K={K}")
        best_ll:         float              = -np.inf
        best_tau:        Optional[np.ndarray] = None
        best_seg_params: Optional[list]      = None
        best_eq_data:    Optional[list]      = None
        best_pi:         Optional[np.ndarray] = None

        for _s in range(n_starts):
            try:
                tau, seg_params, eq_data, pi_k, ll = _run_em(
                    composites, struct_rels, n, K, rng, max_iter, tol
                )
                if ll > best_ll:
                    best_ll        = ll
                    best_tau       = tau
                    best_seg_params = seg_params
                    best_eq_data   = eq_data
                    best_pi        = pi_k
            except Exception as exc:
                warnings.append(f"FIMIX K={K} start failed: {exc}")

        if best_tau is None:
            warnings.append(f"FIMIX K={K}: all starts failed — skipped")
            continue

        # ── Information criteria ──────────────────────────────────────────────
        # free parameters: K × path coefs + (K-1) mixing weights
        n_params = K * n_params_base + (K - 1)
        aic  = round(-2 * best_ll + 2 * n_params, 4)
        bic  = round(-2 * best_ll + np.log(n) * n_params, 4)
        caic = round(-2 * best_ll + (np.log(n) + 1) * n_params, 4)
        r_e  = _relative_entropy(best_tau)

        # ── Build FIMIXSegment objects ────────────────────────────────────────
        hard_assign = np.argmax(best_tau, axis=1)
        min_seg_size = int(hard_assign.shape[0] * 0.01)  # 1% floor
        segment_sizes = np.bincount(hard_assign, minlength=K)
        if np.any(segment_sizes < max(2, min_seg_size)):
            warnings.append(
                f"FIMIX K={K}: degenerate solution — smallest segment has "
                f"{segment_sizes.min()} observations (minimum: {max(2, min_seg_size)}). "
                "Reduce k_max or increase sample size."
            )
            continue
        segs: list[FIMIXSegment] = []

        for k in range(K):
            mask = hard_assign == k
            size = int(mask.sum())
            paths: dict[str, float] = {}
            r2s:   dict[str, float] = {}

            for eq_i, eq in enumerate(best_eq_data):
                coefs = best_seg_params[k][eq_i]["coefs"]

                # Path coefficients
                for rhs_name, c in zip(eq["rhs"], coefs):
                    paths[f"{eq['lhs']}~{rhs_name}"] = round(float(c), 6)

                # Segment-level R² (hard-assigned observations only)
                y_seg = eq["y"][mask]
                x_seg = eq["X"][mask]
                if len(y_seg) > 1:
                    y_hat  = x_seg @ coefs
                    ss_res = np.sum((y_seg - y_hat) ** 2)
                    ss_tot = np.sum((y_seg - y_seg.mean()) ** 2)
                    r2s[eq["lhs"]] = round(
                        float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-14))), 6
                    )

            segs.append(
                FIMIXSegment(
                    segment_id=k + 1,
                    size=size,
                    proportion=round(float(best_pi[k]), 6),
                    path_coefficients=paths,
                    r_squared=r2s,
                )
            )

        solutions.append(
            FIMIXSolution(
                k=K,
                log_likelihood=round(best_ll, 4),
                aic=aic,
                bic=bic,
                caic=caic,
                relative_entropy=r_e,
                segments=segs,
            )
        )

    if not solutions:
        raise ValueError("FIMIX: no solutions converged for any K in [2, k_max].")

    recommended_k = min(solutions, key=lambda s: s.caic).k
    _emit(log_fn, "ok", f"FIMIX complete — recommended K={recommended_k}")

    return FIMIXResult(
        solutions=solutions,
        recommended_k=recommended_k,
        n_obs=n,
        warnings=warnings,
    )
