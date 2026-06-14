"""
engine_mga.py  —  NAVAL-SEM v0.6
=================================
Multi-Group Analysis (MGA) and Higher-Order Construct (HOC) estimation.

Public API
----------
  run_micom()                   : MICOM permutation test (prerequisite for MGA)
  run_mga()                     : Per-group fit + bootstrap path-diff CIs
  fit_hoc_repeated_indicator()  : HOC via repeated-indicator expansion
  fit_hoc_two_stage()           : HOC via two-stage score extraction

All functions follow the existing engine.py conventions:
  - _safe_float() / _emit() for logging and NaN-safety
  - warnings.append() for non-fatal issues
  - try/except with explicit messages on every external call
  - composite-score OLS where semopy is unreliable
"""

from __future__ import annotations

import copy
import logging
import time
from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.engine_utils import _emit, _safe_float
from app.engine import fit_model
from app.parser import (
    build_semopy_syntax,
    detect_hoc,
    expand_hoc_repeated_indicator,
    build_hoc_stage2_parsed,
    parse_lavaan,
)
from app.schemas import (
    FitIndices,
    MGAGroupResult,
    MGAPathDiff,
    MGAResult,
    MICOMResult,
    MICOMStep2Entry,
    MICOMStep3MeanEntry,
    MICOMStep3VarEntry,
    ModelResult,
    PathParameter,
    HOCType,
)

logger = logging.getLogger("naval_sem.mga")


# ── Module-level helpers ───────────────────────────────────────────────────────

def _std_block(df_sub: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Return a mean-0 / sd-1 standardised matrix for the listed columns,
    restricted to rows with no NaN in those columns.
    Falls back to unit-sd if a column has zero variance.
    """
    X = df_sub[cols].astype(float).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def _composite(df_sub: pd.DataFrame, lv: str,
                measurement: dict, weights: dict) -> Optional[np.ndarray]:
    """
    Compute composite scores for `lv` using the given outer-weight dict.
    Returns None when no indicators are found in `df_sub`.
    """
    inds = [i for i in measurement.get(lv, []) if i in df_sub.columns]
    if not inds:
        return None
    X_std = _std_block(df_sub, inds)
    w_lv = weights.get(lv, {})
    w_vec = np.array([w_lv.get(ind, 1.0 / max(len(inds), 1)) for ind in inds])
    return (X_std @ w_vec).astype(float)


def _pls_weights(df: pd.DataFrame, parsed: dict) -> dict[str, dict[str, float]]:
    """
    Fit PLSEstimator and return outer_weights {lv: {ind: w}}.
    Returns empty dict on failure (caller decides how to handle).
    """
    from app.pls import PLSEstimator
    try:
        result = PLSEstimator().fit(df, parsed)
        return result.outer_weights
    except Exception as exc:
        logger.debug("_pls_weights failed: %s", exc)
        return {}


def _pls_paths(df: pd.DataFrame, parsed: dict) -> dict[tuple[str, str], float]:
    """
    Fit PLSEstimator and return {(rhs, lhs): path_coef} for structural paths.
    Key convention: predictor-first (rhs), then outcome (lhs) — consistent
    with _build_coef_map() in engine_utils.py.
    Returns empty dict on failure.
    """
    from app.pls import PLSEstimator
    try:
        result = PLSEstimator().fit(df, parsed)
        coefs: dict[tuple[str, str], float] = {}
        for lhs, rhs_map in result.path_coefficients.items():
            for rhs, coef in rhs_map.items():
                coefs[(rhs, lhs)] = coef
        return coefs
    except Exception as exc:
        logger.debug("_pls_paths failed: %s", exc)
        return {}


def _cb_paths(df: pd.DataFrame, syntax: str) -> dict[tuple[str, str], float]:
    """
    Fit semopy CB-SEM and return {(rhs, lhs): estimate} for structural (~) rows.
    Key convention: predictor-first (rhs), then outcome (lhs) — consistent
    with _build_coef_map() in engine_utils.py.
    Returns empty dict on failure.
    """
    try:
        from semopy import Model
        m = Model(syntax)
        m.fit(df)
        params = m.inspect()
        est_col = "Estimate" if "Estimate" in params.columns else "estimate"
        coefs: dict[tuple[str, str], float] = {}
        for _, row in params.iterrows():
            if str(row.get("op", "")) != "~":
                continue
            lhs = str(row.get("lval", row.get("lhs", "")))
            rhs = str(row.get("rval", row.get("rhs", "")))
            val = _safe_float(row.get(est_col))
            if val is not None:
                coefs[(rhs, lhs)] = val
        return coefs
    except Exception as exc:
        logger.debug("_cb_paths failed: %s", exc)
        return {}


def _fit_group_paths(
    df_g: pd.DataFrame,
    parsed: dict,
    syntax: str,
    algorithm: str,
) -> dict[tuple[str, str], float]:
    """Dispatch to PLS or CB-SEM bootstrap path fitter."""
    if algorithm == "pls":
        return _pls_paths(df_g, parsed)
    return _cb_paths(df_g, syntax)


# ── MICOM ─────────────────────────────────────────────────────────────────────

def run_micom(
    df: pd.DataFrame,
    model_syntax: str,
    group_col: str,
    groups: Optional[list] = None,
    n_permutations: int = 500,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> MICOMResult:
    """
    MICOM — Measurement Invariance of Composites (Henseler, Ringle & Sarstedt 2016).

    Performs a three-step permutation test for measurement invariance across
    exactly two groups using PLS outer weights.

    Step 2 — Compositional invariance
    -----------------------------------
    For each construct c:
      • Fit PLS separately on group 1 (→ w₁) and group 2 (→ w₂).
      • Compute c_obs = cor(X_g1 @ w₁,  X_g1 @ w₂)  using *group-1 data*.
        Under equal weights, both composites are identical → c ≈ 1.
      • Permutation: randomly split all rows into (n₁, n₂) groups, refit PLS
        per split, compute same cross-correlation → build distribution.
      • 5th percentile is the lower CI bound (one-sided test).
      • Invariant when c_obs ≥ ci_lower_95.

    Step 3a — Mean equality
    -------------------------
    Composite scores computed using *pooled-model* weights (fast, no PLS refits).
    Permute group labels → mean-difference distribution → 95 % CI.
    Invariant when 0.0 ∈ [ci_lo, ci_hi].

    Step 3b — Variance equality
    -----------------------------
    Same composites; test whether var(c_g1)/var(c_g2) ≈ 1.
    Invariant when 1.0 ∈ [ci_lo, ci_hi].

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the grouping column.
    model_syntax : str
        lavaan-style model syntax.
    group_col : str
        Column used to split groups.
    groups : list, optional
        Exactly two group values.  Auto-detected from df if None.
    n_permutations : int
        Number of permutation samples (default 500).
    seed : int
        Random seed for reproducibility.
    log_fn : callable, optional
        ``(level, msg) → None`` callback for SSE log streaming.

    Returns
    -------
    MICOMResult
    """
    _emit(log_fn, "step", "MICOM: parsing model and identifying groups")
    parsed = parse_lavaan(model_syntax)
    measurement = parsed.get("measurement", {})
    lvs = list(measurement.keys())

    if not lvs:
        raise ValueError("MICOM requires at least one latent variable with indicators.")

    if groups is None:
        groups = sorted(df[group_col].dropna().unique().tolist())
    groups = list(groups)
    if len(groups) != 2:
        raise ValueError(
            f"MICOM requires exactly 2 groups; found {len(groups)}: {groups}."
        )

    g_a, g_b = groups[0], groups[1]
    df_g1 = df[df[group_col] == g_a].drop(columns=[group_col]).reset_index(drop=True)
    df_g2 = df[df[group_col] == g_b].drop(columns=[group_col]).reset_index(drop=True)
    n1, n2 = len(df_g1), len(df_g2)
    n_total = n1 + n2
    pooled_df = df.drop(columns=[group_col]).reset_index(drop=True)

    _emit(log_fn, "info",
          f"MICOM: group '{g_a}' n={n1} · group '{g_b}' n={n2} · "
          f"{n_permutations} permutations")

    rng = np.random.default_rng(seed)

    # ── Step 2: compositional invariance — per-LV PLS refits needed ──────────
    _emit(log_fn, "step", "MICOM Step 2: compositional invariance (PLS refits per permutation)")
    _t2 = time.time()

    weights_g1 = _pls_weights(df_g1, parsed)
    weights_g2 = _pls_weights(df_g2, parsed)

    def _cross_corr(df_sub: pd.DataFrame, lv: str, w1: dict, w2: dict) -> Optional[float]:
        """cor(X @ w1, X @ w2) on df_sub for construct lv."""
        inds = [i for i in measurement.get(lv, []) if i in df_sub.columns]
        if not inds:
            return None
        X_std = _std_block(df_sub, inds)
        v1 = np.array([w1.get(lv, {}).get(ind, 0.0) for ind in inds])
        v2 = np.array([w2.get(lv, {}).get(ind, 0.0) for ind in inds])
        c1 = X_std @ v1
        c2 = X_std @ v2
        if c1.std() < 1e-12 or c2.std() < 1e-12:
            return None
        r = float(np.corrcoef(c1, c2)[0, 1])
        return None if np.isnan(r) else r

    # Observed Step-2 correlations
    obs_s2 = {lv: _cross_corr(df_g1, lv, weights_g1, weights_g2) for lv in lvs}

    # Permuted Step-2 distributions (requires PLS refits — most expensive part)
    perm_s2: dict[str, list[float]] = {lv: [] for lv in lvs}
    for pi in range(n_permutations):
        if pi > 0 and pi % 100 == 0:
            _emit(log_fn, "info", f"  MICOM Step 2: {pi}/{n_permutations} permutations")
        idx = rng.permutation(n_total)
        dfp1 = pooled_df.iloc[idx[:n1]].reset_index(drop=True)
        dfp2 = pooled_df.iloc[idx[n1:]].reset_index(drop=True)
        wp1 = _pls_weights(dfp1, parsed)
        wp2 = _pls_weights(dfp2, parsed)
        for lv in lvs:
            r = _cross_corr(dfp1, lv, wp1, wp2)
            if r is not None:
                perm_s2[lv].append(r)

    _emit(log_fn, "ok", f"MICOM Step 2 done ({round(time.time() - _t2, 1)}s)")

    step2_entries: list[MICOMStep2Entry] = []
    for lv in lvs:
        c_obs = obs_s2.get(lv)
        if c_obs is None:
            continue
        perm = perm_s2[lv]
        ci_lo = float(np.percentile(perm, 5)) if perm else 0.0
        step2_entries.append(MICOMStep2Entry(
            lv_name=lv,
            correlation=round(c_obs, 6),
            ci_lower_95=round(ci_lo, 6),
            invariant=c_obs >= ci_lo,
        ))

    # ── Step 3: mean/variance equality — pooled weights, fast permute ─────────
    _emit(log_fn, "step", "MICOM Step 3: mean and variance equality (label permutation)")

    weights_pooled = _pls_weights(pooled_df, parsed)

    # Precompute all composites on the full pooled dataset — one matrix op per LV
    all_comp: dict[str, np.ndarray] = {}
    for lv in lvs:
        c = _composite(pooled_df, lv, measurement, weights_pooled)
        if c is not None:
            all_comp[lv] = c

    # Group indicator vector: 0 = group 1, 1 = group 2
    group_labels = np.concatenate([
        np.zeros(n1, dtype=int),
        np.ones(n2,  dtype=int),
    ])

    # Observed Step-3 stats
    obs_mean_diff = {}
    obs_var_ratio  = {}
    for lv, comp in all_comp.items():
        c1, c2 = comp[group_labels == 0], comp[group_labels == 1]
        obs_mean_diff[lv] = float(np.mean(c1) - np.mean(c2))
        v2 = float(np.var(c2, ddof=1))
        obs_var_ratio[lv]  = float(np.var(c1, ddof=1)) / max(v2, 1e-12)

    # Permutation — only label-shuffle needed (very fast)
    perm_mean: dict[str, list[float]] = {lv: [] for lv in all_comp}
    perm_var:  dict[str, list[float]] = {lv: [] for lv in all_comp}
    for _ in range(n_permutations):
        perm_labels = rng.permutation(group_labels)
        for lv, comp in all_comp.items():
            c1 = comp[perm_labels == 0]
            c2 = comp[perm_labels == 1]
            perm_mean[lv].append(float(np.mean(c1) - np.mean(c2)))
            v2p = float(np.var(c2, ddof=1))
            if v2p > 1e-12:
                perm_var[lv].append(float(np.var(c1, ddof=1)) / v2p)

    step3_mean_entries: list[MICOMStep3MeanEntry] = []
    step3_var_entries:  list[MICOMStep3VarEntry]  = []

    for lv in lvs:
        if lv not in all_comp:
            continue
        comp = all_comp[lv]
        c1 = comp[group_labels == 0]
        c2 = comp[group_labels == 1]

        # Mean equality
        md_obs = obs_mean_diff[lv]
        pm = perm_mean[lv]
        m_lo = float(np.percentile(pm, 2.5))  if pm else md_obs
        m_hi = float(np.percentile(pm, 97.5)) if pm else md_obs
        step3_mean_entries.append(MICOMStep3MeanEntry(
            lv_name=lv,
            mean_g1=round(float(np.mean(c1)), 6),
            mean_g2=round(float(np.mean(c2)), 6),
            mean_diff=round(md_obs, 6),
            ci_lower_95=round(m_lo, 6),
            ci_upper_95=round(m_hi, 6),
            invariant=(m_lo <= 0.0 <= m_hi),
        ))

        # Variance equality
        vr_obs = obs_var_ratio[lv]
        pv = perm_var[lv]
        v_lo = float(np.percentile(pv, 2.5))  if pv else vr_obs
        v_hi = float(np.percentile(pv, 97.5)) if pv else vr_obs
        step3_var_entries.append(MICOMStep3VarEntry(
            lv_name=lv,
            var_g1=round(float(np.var(c1, ddof=1)), 6),
            var_g2=round(float(np.var(c2, ddof=1)), 6),
            var_ratio=round(vr_obs, 6),
            ci_lower_95=round(v_lo, 6),
            ci_upper_95=round(v_hi, 6),
            invariant=(v_lo <= 1.0 <= v_hi),
        ))

    full_inv = (
        all(e.invariant for e in step2_entries)
        and all(e.invariant for e in step3_mean_entries)
        and all(e.invariant for e in step3_var_entries)
    )
    partial_inv = all(e.invariant for e in step2_entries)

    _emit(log_fn, "ok",
          f"MICOM complete — partial invariance: {partial_inv} · "
          f"full invariance: {full_inv}")

    return MICOMResult(
        n_permutations=n_permutations,
        groups=[str(g_a), str(g_b)],
        step2=step2_entries,
        step3_mean=step3_mean_entries,
        step3_var=step3_var_entries,
        full_invariance=full_inv,
        partial_invariance=partial_inv,
    )


# ── MGA ───────────────────────────────────────────────────────────────────────

def run_mga(
    df: pd.DataFrame,
    model_syntax: str,
    group_col: str,
    algorithm: str = "pls",
    bootstrap_n: int = 500,
    n_permutations: int = 500,
    run_micom_test: bool = True,
    log_fn: Optional[Callable] = None,
    seed: int = 42,
) -> MGAResult:
    """
    Multi-Group Analysis.

    Procedure
    ---------
    1. [Optional] Run MICOM (for 2-group analyses only).
    2. For each group: fit the full model with ``fit_model()`` (bootstrap_n=0
       to keep individual group fits fast; path significance comes from the
       MGA bootstrap in step 3).
    3. For every pair of groups and every structural path:
       - Observed difference: β_a − β_b from the full-data per-group fits.
       - Bootstrap: independently resample each group's data ``bootstrap_n``
         times, refit PLS/CB-SEM per group per sample, collect β differences.
       - 95 % percentile CI from the bootstrap distribution.
       - Significant when CI excludes 0.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (must contain ``group_col``).
    model_syntax : str
        lavaan-style syntax (same model applied to every group).
    group_col : str
        Grouping column.  Values are stringified; up to 10 groups supported.
    algorithm : str
        ``"pls"`` (default) | ``"cb"`` | ``"wls"``.
    bootstrap_n : int
        Bootstrap samples for path-difference CIs (default 500).
    n_permutations : int
        Permutations passed to ``run_micom()`` when ``run_micom_test=True``.
    run_micom_test : bool
        Whether to run MICOM before MGA (strongly recommended).
    log_fn : callable, optional
        SSE logging callback.
    seed : int
        Random seed.

    Returns
    -------
    MGAResult
    """
    _emit(log_fn, "step", "MGA: parsing model syntax")
    parsed   = parse_lavaan(model_syntax)
    syntax   = build_semopy_syntax(parsed)
    structural = parsed.get("structural", [])
    warnings: list[str] = []

    if not structural:
        raise ValueError("MGA requires at least one structural path.")

    # ── Identify groups ───────────────────────────────────────────────────────
    if group_col not in df.columns:
        raise ValueError(
            f"Grouping column '{group_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )
    groups = sorted(df[group_col].dropna().unique().tolist())
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(f"MGA requires ≥ 2 groups; found {n_groups}.")
    if n_groups > 10:
        raise ValueError(
            f"MGA found {n_groups} groups — max is 10 to prevent runaway compute."
        )
    _emit(log_fn, "info",
          f"MGA: {n_groups} groups — {[str(g) for g in groups]}")

    # ── Step 1: MICOM (2-group only) ──────────────────────────────────────────
    micom_result: Optional[MICOMResult] = None
    if run_micom_test and n_groups == 2 and algorithm == "pls":
        try:
            _emit(log_fn, "step",
                  f"MGA → MICOM ({n_permutations} permutations)")
            micom_result = run_micom(
                df, model_syntax, group_col,
                groups=groups,
                n_permutations=n_permutations,
                seed=seed,
                log_fn=log_fn,
            )
            if not micom_result.partial_invariance:
                warnings.append(
                    "MICOM Step 2 failed for ≥ 1 construct — path-coefficient "
                    "comparisons may be invalid.  Interpret MGA results with caution."
                )
        except Exception as exc:
            warnings.append(f"MICOM could not be computed: {exc}")
    elif run_micom_test and algorithm != "pls":
        warnings.append(
            "MICOM is only implemented for algorithm='pls'. "
            "Skipping measurement invariance test."
        )

    # ── Step 2: Per-group model fits ──────────────────────────────────────────
    _emit(log_fn, "step", "MGA: fitting model per group")
    group_results: list[MGAGroupResult] = []
    group_path_maps: dict[str, dict[tuple[str, str], float]] = {}

    for g in groups:
        g_str = str(g)
        df_g = df[df[group_col] == g].drop(columns=[group_col])
        n_g  = len(df_g)
        _emit(log_fn, "info", f"  Fitting group '{g_str}' (n={n_g})")

        if n_g < max(10, len(parsed.get("observed_vars", [])) + 2):
            warnings.append(
                f"Group '{g_str}' has only {n_g} observations — "
                "estimates may be unstable."
            )
        try:
            res_g = fit_model(
                df_g, model_syntax,
                algorithm=algorithm,
                bootstrap_n=0,   # per-group significance comes from MGA bootstrap
                log_fn=None,     # suppress per-group sub-logs
            )
        except Exception as exc:
            warnings.append(f"Group '{g_str}' model fit failed: {exc}")
            continue

        # Collect path coefs for bootstrap diff computation
        group_path_maps[g_str] = {
            (p.rhs, p.lhs): p.estimate
            for p in res_g.parameters
            if p.op == "~"
        }

        group_results.append(MGAGroupResult(
            group_name=g_str,
            n_obs=res_g.n_obs,
            parameters=res_g.parameters,
            fit=res_g.fit,
            r_squared=res_g.fit.r_squared,
        ))

    if len(group_results) < 2:
        raise ValueError(
            "MGA: fewer than 2 groups produced valid model fits. "
            "Check warnings for details."
        )

    # ── Step 3: Bootstrap path-difference CIs ─────────────────────────────────
    _emit(log_fn, "step",
          f"MGA: bootstrap path differences ({bootstrap_n} samples per group pair)")
    rng = np.random.default_rng(seed)
    path_differences: list[MGAPathDiff] = []
    _t_bs = time.time()

    import itertools
    n_pairs = len(list(itertools.combinations(groups, 2)))
    estimated_fits = n_pairs * bootstrap_n * 2
    FIT_LIMIT = 500_000
    if estimated_fits > FIT_LIMIT:
        raise ValueError(
            f"MGA: estimated {estimated_fits:,} bootstrap model fits "
            f"({n_pairs} pairs x {bootstrap_n} iterations x 2) exceeds "
            f"the safety limit of {FIT_LIMIT:,}. "
            "Reduce bootstrap_n or the number of groups."
        )

    for (i_a, g_a), (i_b, g_b) in combinations(enumerate(groups), 2):
        g_a_str, g_b_str = str(g_a), str(g_b)
        df_a = df[df[group_col] == g_a].drop(columns=[group_col]).reset_index(drop=True)
        df_b = df[df[group_col] == g_b].drop(columns=[group_col]).reset_index(drop=True)

        coef_a_obs = group_path_maps.get(g_a_str, {})
        coef_b_obs = group_path_maps.get(g_b_str, {})

        # Bootstrap distribution of β_a − β_b for each structural path
        bs_diffs: dict[tuple[str, str], list[float]] = {
            (rel["rhs"], rel["lhs"]): [] for rel in structural
        }
        converged_bs = 0

        for bi in range(bootstrap_n):
            if bi > 0 and bi % 100 == 0:
                _emit(log_fn, "info",
                      f"  MGA bootstrap ({g_a_str} vs {g_b_str}): "
                      f"{bi}/{bootstrap_n} · {converged_bs} converged")
            sample_a = df_a.sample(len(df_a), replace=True,
                                   random_state=int(rng.integers(1_000_000)))
            sample_b = df_b.sample(len(df_b), replace=True,
                                   random_state=int(rng.integers(1_000_000)))
            try:
                paths_a = _fit_group_paths(sample_a, parsed, syntax, algorithm)
                paths_b = _fit_group_paths(sample_b, parsed, syntax, algorithm)
                ok = False
                for rel in structural:
                    key = (rel["rhs"], rel["lhs"])
                    ba = paths_a.get(key)
                    bb = paths_b.get(key)
                    if ba is not None and bb is not None:
                        bs_diffs[key].append(ba - bb)
                        ok = True
                if ok:
                    converged_bs += 1
            except Exception as exc:
                logger.debug("MGA bootstrap sample failed: %s", exc)
                continue

        _emit(log_fn, "info",
              f"  MGA bootstrap ({g_a_str} vs {g_b_str}): "
              f"{converged_bs}/{bootstrap_n} samples converged")
        if converged_bs == 0:
            warnings.append(
                f"No bootstrap samples converged for pair "
                f"'{g_a_str}' vs '{g_b_str}'. CIs set to [0, 0]."
            )

        # Assemble MGAPathDiff entries for this group pair
        for rel in structural:
            lhs, rhs = rel["lhs"], rel["rhs"]
            key = (rhs, lhs)
            beta_a = coef_a_obs.get(key, 0.0)
            beta_b = coef_b_obs.get(key, 0.0)
            diff_obs = beta_a - beta_b

            bs = bs_diffs.get(key, [])
            if len(bs) >= 10:
                ci_lo = float(np.percentile(bs, 2.5))
                ci_hi = float(np.percentile(bs, 97.5))
            else:
                ci_lo = ci_hi = diff_obs
                warnings.append(
                    f"Too few bootstrap samples for path {lhs}~{rhs} "
                    f"({g_a_str} vs {g_b_str}) — CI set to point estimate."
                )

            path_differences.append(MGAPathDiff(
                lhs=lhs,
                rhs=rhs,
                group_a=g_a_str,
                group_b=g_b_str,
                beta_a=round(beta_a, 6),
                beta_b=round(beta_b, 6),
                diff=round(diff_obs, 6),
                ci_lower_95=round(ci_lo, 6),
                ci_upper_95=round(ci_hi, 6),
                significant=not (ci_lo <= 0.0 <= ci_hi),
            ))

    _emit(log_fn, "ok",
          f"MGA complete — {len(path_differences)} path-difference CI(s) · "
          f"{round(time.time() - _t_bs, 1)}s bootstrap")

    return MGAResult(
        grouping_variable=group_col,
        groups=[str(g) for g in groups],
        bootstrap_n=bootstrap_n,
        group_results=group_results,
        path_differences=path_differences,
        micom=micom_result,
        warnings=warnings,
    )


# ── HOC: Repeated Indicator ───────────────────────────────────────────────────

def fit_hoc_repeated_indicator(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
    bootstrap_n: int = 0,
    log_fn: Optional[Callable] = None,
) -> ModelResult:
    """
    Estimate a Higher-Order Construct model via the **repeated indicator**
    approach (Wold 1982; Lohmöller 1989; Hair et al. 2022).

    Each HOC's measurement block is expanded to include all indicators from
    its constituent First-Order Constructs (FOCs).  The FOC measurement blocks
    are retained so both levels are estimated jointly.

    Example lavaan syntax::

        HOC  =~ FOC1 + FOC2       # HOC uses FOC names as "indicators"
        FOC1 =~ x1 + x2 + x3
        FOC2 =~ x4 + x5 + x6
        Y    ~  HOC               # structural paths unchanged

    Falls through to ``fit_model()`` if no HOCs are detected.

    Returns
    -------
    ModelResult
        ``hoc_type`` is set to ``HOCType.repeated_indicator``.
        A warning is prepended describing the expansion.
    """
    _emit(log_fn, "step", "HOC repeated-indicator: parsing model")
    parsed_orig = parse_lavaan(model_syntax)
    hoc_map     = detect_hoc(parsed_orig)

    if not hoc_map:
        _emit(log_fn, "info",
              "HOC repeated-indicator: no higher-order constructs detected — "
              "running standard fit_model()")
        return fit_model(df, model_syntax,
                         algorithm=algorithm,
                         bootstrap_n=bootstrap_n,
                         log_fn=log_fn)

    _emit(log_fn, "info",
          f"HOC detected: {list(hoc_map.keys())} "
          f"← {list(hoc_map.values())}")

    # Expand HOC measurement blocks to include all FOC indicators
    parsed_expanded = expand_hoc_repeated_indicator(parsed_orig)
    syntax_expanded = build_semopy_syntax(parsed_expanded)

    _emit(log_fn, "step",
          "HOC repeated-indicator: fitting expanded model")
    result = fit_model(df, syntax_expanded,
                       algorithm=algorithm,
                       bootstrap_n=bootstrap_n,
                       log_fn=log_fn)

    result.hoc_type = HOCType.repeated_indicator
    result.warnings.insert(0,
        f"HOC repeated-indicator: indicators of "
        f"{list(hoc_map.keys())} expanded to include all FOC indicators "
        f"({list(hoc_map.values())})."
    )
    return result


# ── HOC: Two-Stage ────────────────────────────────────────────────────────────

def fit_hoc_two_stage(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
    bootstrap_n: int = 0,
    log_fn: Optional[Callable] = None,
) -> ModelResult:
    """
    Estimate a Higher-Order Construct model via the **two-stage** approach
    (Ringle, Sarstedt & Straub 2012).

    Stage 1
    -------
    Fit a PLS model containing *only* the First-Order Constructs (FOCs) and
    any structural paths among them (not involving the HOC).  Extract LV
    composite scores for every FOC that serves as a HOC indicator.

    Stage 2
    -------
    Add the Stage-1 LV scores as new data columns (named ``__score_<FOC>__``).
    Replace the HOC's FOC-name indicators with these score column names.
    Remove the FOC measurement blocks from the Stage-2 model (the FOCs are
    now treated as observed variables via their scores).
    Fit the full structural model (including the HOC) on the augmented dataset.

    Limitations
    -----------
    * Currently implemented for PLS-SEM only (Stage 1 always uses PLS).
      Stage 2 respects the ``algorithm`` argument.
    * CB-SEM is *not* supported for Stage 1 (semopy's factor scores are
      unreliable for this purpose).

    Falls through to ``fit_model()`` if no HOCs are detected.

    Returns
    -------
    ModelResult
        ``hoc_type`` is set to ``HOCType.two_stage``.
    """
    from app.pls import PLSEstimator

    _emit(log_fn, "step", "HOC two-stage: parsing model")
    parsed_orig = parse_lavaan(model_syntax)
    hoc_map     = detect_hoc(parsed_orig)

    if not hoc_map:
        _emit(log_fn, "info",
              "HOC two-stage: no higher-order constructs detected — "
              "running standard fit_model()")
        return fit_model(df, model_syntax,
                         algorithm=algorithm,
                         bootstrap_n=bootstrap_n,
                         log_fn=log_fn)

    _emit(log_fn, "info",
          f"HOC detected: {list(hoc_map.keys())} ← {list(hoc_map.values())}")

    measurement = parsed_orig.get("measurement", {})
    hoc_set     = set(hoc_map.keys())
    foc_set     = {foc for focs in hoc_map.values() for foc in focs}

    # ── Stage 1: FOC-only model ────────────────────────────────────────────────
    stage1_measurement = {
        lv: inds for lv, inds in measurement.items()
        if lv not in hoc_set
    }
    # Keep only structural paths that do NOT involve any HOC
    stage1_structural = [
        rel for rel in parsed_orig.get("structural", [])
        if rel["lhs"] not in hoc_set and rel["rhs"] not in hoc_set
    ]
    stage1_parsed = {
        "measurement":  stage1_measurement,
        "structural":   stage1_structural,
        "covariances":  [],
        "latent_vars":  list(stage1_measurement.keys()),
        "observed_vars": list({
            v for inds in stage1_measurement.values() for v in inds
        }),
    }

    _emit(log_fn, "step",
          f"HOC two-stage — Stage 1: fitting FOC model "
          f"({list(stage1_measurement.keys())})")
    try:
        pls_s1 = PLSEstimator().fit(df, stage1_parsed)
    except Exception as exc:
        raise ValueError(f"HOC two-stage Stage 1 PLS failed: {exc}") from exc

    _emit(log_fn, "info",
          f"  Stage 1 converged in {pls_s1.n_iterations} iterations "
          f"(n_obs={pls_s1.n_obs})")

    # ── Inject Stage-1 scores as observed columns ─────────────────────────────
    # Only FOCs that are HOC indicators need score columns.
    df_s2 = df.copy()
    score_col_map: dict[str, str] = {}
    for foc in foc_set:
        if foc in pls_s1.scores.columns:
            col_name = f"__score_{foc}__"
            df_s2[col_name] = pls_s1.scores[foc].values
            score_col_map[foc] = col_name
        else:
            _emit(log_fn, "warn",
                  f"  Stage 1 did not produce scores for FOC '{foc}' — "
                  "it will be skipped in Stage 2.")

    if not score_col_map:
        raise ValueError(
            "HOC two-stage: Stage 1 produced no LV scores for any FOC. "
            "Check that your FOCs have valid measurement blocks."
        )

    # ── Build Stage-2 parsed dict ──────────────────────────────────────────────
    _emit(log_fn, "step",
          f"HOC two-stage — Stage 2: building model with score columns "
          f"{list(score_col_map.values())}")
    try:
        stage2_parsed = build_hoc_stage2_parsed(parsed_orig, score_col_map)
    except Exception as exc:
        raise ValueError(f"HOC two-stage Stage 2 model build failed: {exc}") from exc

    stage2_syntax = build_semopy_syntax(stage2_parsed)

    # Validate all required columns are in df_s2
    missing = [
        v for v in stage2_parsed["observed_vars"]
        if v not in df_s2.columns
    ]
    if missing:
        raise ValueError(
            f"HOC two-stage Stage 2: columns not found in augmented dataset: {missing}"
        )

    _emit(log_fn, "step",
          "HOC two-stage — Stage 2: fitting full model")
    result = fit_model(df_s2, stage2_syntax,
                       algorithm=algorithm,
                       bootstrap_n=bootstrap_n,
                       log_fn=log_fn)

    result.hoc_type = HOCType.two_stage
    result.warnings.insert(0,
        f"HOC two-stage: Stage-1 PLS scores used as indicators for "
        f"{list(hoc_map.keys())}.  FOC measurement blocks removed from Stage 2. "
        f"Score columns: {list(score_col_map.values())}."
    )
    return result
