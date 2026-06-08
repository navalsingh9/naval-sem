"""
engine_mod_mediation.py  —  NAVAL-SEM v0.7
===========================================
Moderated Mediation (Conditional Process Analysis).

Edwards & Lambert (2007); Hayes (2018, Introduction to Mediation,
Moderation, and Conditional Process Analysis, Chapters 11–14).

Public API
----------
  run_mod_mediation(df, model_syntax, algorithm, bootstrap_n, seed, log_fn)
    -> ModMediationResult

Supported patterns
------------------
  a-path moderation — W moderates X → M  (Hayes Process Model 7):
      Y ~ X + M
      M ~ X + W + X*W
      X =~ ...   M =~ ...   Y =~ ...   W =~ ...

  b-path moderation — W moderates M → Y  (Hayes Process Model 14):
      Y ~ X + M + W + M*W
      M ~ X
      ...

  Both paths — W moderates both X→M and M→Y  (Hayes Process Model 58/59):
      Y ~ X + M + W + M*W
      M ~ X + W + X*W
      ...

Index of Moderated Mediation (IMM)
-----------------------------------
  a-path only:  IMM = a₃ × b        (evaluated at mean W = 0)
  b-path only:  IMM = a × b₃
  both paths:   Reported per-path; overall conditional IE at W ± 1 SD
                captures the curvilinear relationship.

Conditional indirect effects at W = −1 SD, mean (0), +1 SD
-----------------------------------------------------------
  a-path: IE(w) = (a + a₃·w) × b
  b-path: IE(w) = a × (b + b₃·w)
  both:   IE(w) = (a + a₃·w) × (b + b₃·w)
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

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
    ConditionalIndirectEffect,
    FitIndices,
    ModMediationPath,
    ModMediationResult,
    PathParameter,
)

logger = logging.getLogger("naval_sem.mod_mediation")


# ── private helpers ────────────────────────────────────────────────────────────

def _coef_from_params(
    parameters: list[PathParameter],
    lhs: str,
    rhs: str,
) -> Optional[float]:
    """Return the fitted estimate for a specific (lhs ~ rhs) path, or None."""
    for p in parameters:
        if p.op == "~" and p.lhs == lhs and p.rhs == rhs:
            return p.estimate
    return None


def _ci_from_bootstrap(
    samples: list[float],
) -> tuple[Optional[float], Optional[float]]:
    """Return 95 % percentile CI from bootstrap distribution (min 10 samples)."""
    if len(samples) < 10:
        return None, None
    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def _sig(lo: Optional[float], hi: Optional[float]) -> bool:
    """True when 95 % CI excludes zero."""
    if lo is None or hi is None:
        return False
    return not (lo <= 0.0 <= hi)


# ── chain detection ────────────────────────────────────────────────────────────

def _build_structural_graph(
    structural: list[dict],
    exclude_rhs: set[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Build forward (from → [to]) and reverse (to → [from]) adjacency from
    clean structural paths, excluding interaction product columns.

    Returns (forward, reverse).
    """
    forward: dict[str, list[str]] = {}
    reverse: dict[str, list[str]] = {}
    for rel in structural:
        rhs, lhs = rel["rhs"], rel["lhs"]
        if "*" in rhs or rhs in exclude_rhs:
            continue
        forward.setdefault(rhs, []).append(lhs)
        reverse.setdefault(lhs, []).append(rhs)
    return forward, reverse


# ── public API ─────────────────────────────────────────────────────────────────

def run_mod_mediation(
    df: pd.DataFrame,
    model_syntax: str,
    algorithm: str = "pls",
    bootstrap_n: int = 500,
    seed: int = 42,
    log_fn: Optional[Callable] = None,
) -> ModMediationResult:
    """
    Moderated Mediation / Conditional Process Analysis.

    Detects interaction terms in the lavaan structural block and infers
    the mediation chain (X → M → Y) for each one. For every detected
    chain it computes:

    1. Direct path coefficients (a, b, c', interaction).
    2. Index of Moderated Mediation (IMM) with bootstrap 95 % CI.
    3. Conditional indirect effects at moderator W = −1 SD, 0, +1 SD
       with bootstrap 95 % CI.

    Parameters
    ----------
    df           : pd.DataFrame
    model_syntax : str   lavaan syntax with at least one ``X*W`` interaction.
    algorithm    : str   ``"pls"`` | ``"cb"`` | ``"wls"``.
    bootstrap_n  : int   Bootstrap resamples for CIs (0 = skip bootstrapping).
    seed         : int
    log_fn       : callable | None

    Returns
    -------
    ModMediationResult
    """
    _emit(log_fn, "step",
          "ModMediation: parsing syntax and detecting interactions")

    parsed_orig  = parse_lavaan(model_syntax)
    interactions = detect_interactions(parsed_orig)

    if not interactions:
        raise ValueError(
            "No interaction terms detected in the model syntax. "
            "Moderated mediation requires at least one interaction term using "
            "'X*W' notation in a structural path.  Examples:\n"
            "  M ~ X + W + X*W          ← W moderates the a-path (X→M)\n"
            "  Y ~ X + M + W + M*W      ← W moderates the b-path (M→Y)"
        )

    _emit(log_fn, "info",
          f"  {len(interactions)} interaction term(s) detected: "
          + ", ".join(f"{i['iv']}*{i['moderator']}" for i in interactions))

    structural   = parsed_orig.get("structural", [])
    measurement  = parsed_orig.get("measurement", {})
    warnings:    list[str] = []
    rng = np.random.default_rng(seed)

    # Interaction column names — excluded from the "real" structural graph
    interaction_cols = {i["interaction_col"] for i in interactions}

    # Build plain structural graph (no interaction cols, no * terms)
    forward_graph, reverse_graph = _build_structural_graph(
        structural, interaction_cols
    )

    # ── Expand interaction terms ───────────────────────────────────────────────
    try:
        parsed_full, df_aug = expand_interaction_terms(parsed_orig, df)
    except ValueError as exc:
        raise ValueError(f"ModMediation setup failed: {exc}") from exc

    syntax_full = build_semopy_syntax(parsed_full)

    # ── Fit full model ─────────────────────────────────────────────────────────
    _emit(log_fn, "step", "ModMediation: fitting full model")
    try:
        res_full = fit_model(
            df_aug, syntax_full,
            algorithm=algorithm, bootstrap_n=0, log_fn=None,
        )
    except Exception as exc:
        raise ValueError(f"ModMediation full-model fit failed: {exc}") from exc

    params = res_full.parameters

    # ── Moderator composite for SD calculation ─────────────────────────────────
    all_composites = _build_composites(df, measurement, structural)

    def _mod_sd(w_var: str) -> float:
        comp = all_composites.get(w_var)
        if comp is None and w_var in df.columns:
            comp = df[w_var].astype(float)
        return float(comp.std()) if comp is not None else 1.0

    # ── Resolve X, M, Y, W and path type for each interaction ─────────────────
    mm_paths: list[ModMediationPath] = []

    for itx in interactions:
        iv            = itx["iv"]            # variable that interacts (X or M)
        moderator     = itx["moderator"]     # W
        outcome_itx   = itx["outcome"]       # LHS of the *-path structural row
        icol          = itx["interaction_col"]

        # ── Determine a-path vs b-path ────────────────────────────────────────
        #
        # a-path: outcome_itx is a MEDIATOR → it has downstream effects
        #   Syntax:  M ~ X + W + X*W         (iv=X, outcome=M)
        #
        # b-path: outcome_itx is the FINAL DV → nothing downstream
        #   Syntax:  Y ~ X + M + W + M*W     (iv=M, outcome=Y)
        #
        downstream = [
            v for v in forward_graph.get(outcome_itx, [])
            if v not in interaction_cols and v != moderator
        ]

        if downstream:
            # ── a-path moderation ─────────────────────────────────────────────
            moderated_path = "a"
            x_var = iv
            m_var = outcome_itx

            # Pick Y: first downstream of M that x_var also directly predicts
            y_var = downstream[0]
            if len(downstream) > 1:
                for cand in downstream:
                    if any(
                        rel["lhs"] == cand and rel["rhs"] == x_var
                        for rel in structural
                        if "*" not in rel.get("rhs", "")
                    ):
                        y_var = cand
                        break

        else:
            # ── b-path moderation ─────────────────────────────────────────────
            moderated_path = "b"
            y_var = outcome_itx
            m_var = iv   # the interacting variable IS the mediator

            # Find X: predictor of M that is not W
            x_candidates = [
                pred for pred in reverse_graph.get(m_var, [])
                if pred != moderator and pred not in interaction_cols
            ]
            if not x_candidates:
                warnings.append(
                    f"ModMediation: could not find a predictor of '{m_var}' "
                    f"that is not the moderator '{moderator}' — "
                    f"interaction {itx['term']} skipped."
                )
                continue
            x_var = x_candidates[0]

        _emit(log_fn, "step",
              f"  Chain: {x_var} → {m_var} → {y_var}  "
              f"(W={moderator}, moderated: {moderated_path}-path)")

        # ── Extract point-estimate coefficients ───────────────────────────────
        a_path  = _safe_float(_coef_from_params(params, m_var, x_var))  or 0.0
        b_path  = _safe_float(_coef_from_params(params, y_var, m_var))  or 0.0
        c_prime = _safe_float(_coef_from_params(params, y_var, x_var))  or 0.0

        a3_int: Optional[float] = None
        b3_int: Optional[float] = None

        if moderated_path == "a":
            a3_int = _safe_float(_coef_from_params(params, m_var, icol))
            if a3_int is None:
                warnings.append(
                    f"ModMediation: interaction coefficient for '{icol}→{m_var}' "
                    "not found in fitted parameters; set to 0."
                )
                a3_int = 0.0
        else:
            b3_int = _safe_float(_coef_from_params(params, y_var, icol))
            if b3_int is None:
                warnings.append(
                    f"ModMediation: interaction coefficient for '{icol}→{y_var}' "
                    "not found in fitted parameters; set to 0."
                )
                b3_int = 0.0

        # ── Moderator levels for conditional IEs ──────────────────────────────
        mod_sd     = _mod_sd(moderator)
        mod_levels = [-mod_sd, 0.0, mod_sd]

        # Point-estimate conditional indirect effects
        def _ie_point(lvl: float) -> float:
            if moderated_path == "a":
                return (a_path + (a3_int or 0.0) * lvl) * b_path
            else:
                return a_path * (b_path + (b3_int or 0.0) * lvl)

        # Point-estimate IMM (at mean, i.e. W=0)
        if moderated_path == "a":
            imm_pt = (a3_int or 0.0) * b_path
        else:
            imm_pt = a_path * (b3_int or 0.0)

        # ── Bootstrap ─────────────────────────────────────────────────────────
        if bootstrap_n > 0:
            _emit(log_fn, "step",
                  f"  Bootstrap ({bootstrap_n} samples) for {x_var}×{moderator}")

        bs_imm:  list[float]       = []
        bs_cond: list[list[float]] = [[] for _ in mod_levels]

        for _bi in range(bootstrap_n):
            try:
                idx   = rng.integers(0, len(df_aug), size=len(df_aug))
                df_bs = df_aug.iloc[idx].reset_index(drop=True)
                r_bs  = fit_model(df_bs, syntax_full,
                                  algorithm=algorithm, bootstrap_n=0)

                a_bs = _coef_from_params(r_bs.parameters, m_var, x_var) or 0.0
                b_bs = _coef_from_params(r_bs.parameters, y_var, m_var) or 0.0

                if moderated_path == "a":
                    a3_bs = _coef_from_params(r_bs.parameters, m_var, icol) or 0.0
                    bs_imm.append(a3_bs * b_bs)
                    for k, lvl in enumerate(mod_levels):
                        bs_cond[k].append((a_bs + a3_bs * lvl) * b_bs)
                else:
                    b3_bs = _coef_from_params(r_bs.parameters, y_var, icol) or 0.0
                    bs_imm.append(a_bs * b3_bs)
                    for k, lvl in enumerate(mod_levels):
                        bs_cond[k].append(a_bs * (b_bs + b3_bs * lvl))

            except Exception:
                continue

        imm_lo, imm_hi = _ci_from_bootstrap(bs_imm)

        # ── Build conditional-effect entries ───────────────────────────────────
        level_labels = ["low (−1 SD)", "mean (0)", "high (+1 SD)"]
        cond_effects: list[ConditionalIndirectEffect] = []

        for k, (label, lvl) in enumerate(zip(level_labels, mod_levels)):
            cie_lo, cie_hi = _ci_from_bootstrap(bs_cond[k])
            cond_effects.append(ConditionalIndirectEffect(
                moderator_level=label,
                moderator_value=round(float(lvl), 4),
                indirect_effect=round(float(_ie_point(lvl)), 6),
                ci_lower_95=round(cie_lo, 6) if cie_lo is not None else None,
                ci_upper_95=round(cie_hi, 6) if cie_hi is not None else None,
                significant=_sig(cie_lo, cie_hi),
            ))

        mm_paths.append(ModMediationPath(
            x=x_var, m=m_var, y=y_var, w=moderator,
            moderated_path=moderated_path,
            a_path=round(a_path, 6),
            b_path=round(b_path, 6),
            c_prime=round(c_prime, 6),
            a3_interaction=round(float(a3_int), 6) if a3_int is not None else None,
            b3_interaction=round(float(b3_int), 6) if b3_int is not None else None,
            imm=round(float(imm_pt), 6),
            imm_ci_lower_95=round(imm_lo, 6) if imm_lo is not None else None,
            imm_ci_upper_95=round(imm_hi, 6) if imm_hi is not None else None,
            imm_significant=_sig(imm_lo, imm_hi),
            conditional_effects=cond_effects,
        ))

        _emit(log_fn, "ok",
              f"  {x_var}→{m_var}→{y_var}: "
              f"IMM={imm_pt:.4f}  sig={_sig(imm_lo, imm_hi)}")

    if not mm_paths:
        raise ValueError(
            "ModMediation: no valid X→M→Y chain could be resolved from the "
            "model syntax. Ensure there is at least one complete mediation "
            "path and that all variable names match the dataset columns."
        )

    _emit(log_fn, "ok",
          f"ModMediation complete — {len(mm_paths)} chain(s) estimated")

    return ModMediationResult(
        algorithm=algorithm,
        n_obs=res_full.n_obs,
        bootstrap_n=bootstrap_n,
        paths=mm_paths,
        parameters=res_full.parameters,
        fit=res_full.fit,
        warnings=warnings + res_full.warnings,
    )
