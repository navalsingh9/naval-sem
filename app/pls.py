"""
app/pls.py
==========
Pure-numpy PLS-SEM estimator for NAVAL-SEM.

Implements the standard PLSPM iterative algorithm
(Lohmöller 1989 / Hair, Ringle & Sarstedt 2022):

  • Mode A (reflective) outer weight estimation
  • Path weighting scheme for inner approximation
  • Convergence loop (max 300 iterations, ε = 1e-7)
  • Structural OLS for path coefficients and R²
  • SRMR from model-implied correlation matrix

Returns PLSResult — consumed by engine.fit_model().
No external dependencies beyond numpy / pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("naval_sem.pls")
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PLSResult:
    """
    All outputs from a single PLS-SEM run.
    Designed so engine.py can consume this without schema changes.
    """
    scores:            pd.DataFrame                    # LV scores  (n_obs × n_lv)
    outer_weights:     Dict[str, Dict[str, float]]     # {lv: {indicator: weight}}
    outer_loadings:    Dict[str, Dict[str, float]]     # {lv: {indicator: loading}}
    path_coefficients: Dict[str, Dict[str, float]]     # {lhs: {rhs: coef}}  structural
    r_squared:         Dict[str, float]                # {endogenous_lv: R²}
    srmr:              Optional[float]                 # model-implied correlation SRMR
    n_iterations:      int
    converged:         bool
    n_obs:             int
    warnings:          List[str] = field(default_factory=list)


# ── Estimator ─────────────────────────────────────────────────────────────────

class PLSEstimator:
    """
    PLS-SEM estimator.

    Usage
    -----
    result = PLSEstimator().fit(df, parsed)

    Parameters
    ----------
    df     : pandas DataFrame with indicator columns
    parsed : dict from parse_lavaan() — keys: measurement, structural,
             latent_vars, observed_vars
    """

    def __init__(
        self,
        max_iter:  int   = 300,
        tolerance: float = 1e-7,
        scheme:    str   = "path",   # "path" | "centroid"
    ):
        self.max_iter  = max_iter
        self.tolerance = tolerance
        self.scheme    = scheme

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, parsed: dict) -> PLSResult:
        """Run PLS-SEM and return a PLSResult."""

        measurement: Dict[str, List[str]] = parsed.get("measurement", {})
        structural:  List[dict]           = parsed.get("structural",  [])
        lvs:         List[str]            = list(measurement.keys())
        warnings:    List[str]            = []

        if not lvs:
            raise ValueError(
                "PLS-SEM requires at least one latent variable with indicators."
            )

        # ── 1. Standardise indicator matrix ───────────────────────────────────
        # Only keep indicators present in df; standardise to mean=0, sd=1.
        all_indicators = [
            ind for inds in measurement.values()
            for ind in inds if ind in df.columns
        ]
        if not all_indicators:
            raise ValueError(
                "None of the model indicators were found in the data columns."
            )

        X_raw = df[all_indicators].astype(float)
        means = X_raw.mean()
        stds  = X_raw.std(ddof=1).replace(0.0, 1.0)
        X_std = ((X_raw - means) / stds).values   # (n × p)

        n_obs   = X_std.shape[0]
        ind_idx = {ind: i for i, ind in enumerate(all_indicators)}

        # ── 2. Build graph helpers ─────────────────────────────────────────────
        predecessors: Dict[str, List[str]] = {lv: [] for lv in lvs}
        successors:   Dict[str, List[str]] = {lv: [] for lv in lvs}
        for rel in structural:
            lhs, rhs = rel["lhs"], rel["rhs"]
            if lhs in predecessors and rhs in successors:
                predecessors[lhs].append(rhs)
                successors[rhs].append(lhs)

        lv_idx = {lv: j for j, lv in enumerate(lvs)}

        # ── 3. Initialise outer weights (equal, normalised) ────────────────────
        weights: Dict[str, np.ndarray] = {}
        for lv, inds in measurement.items():
            k = sum(1 for ind in inds if ind in ind_idx)
            weights[lv] = np.ones(k) / max(np.sqrt(k), 1e-12)

        # ── 4. Iterative PLS algorithm ─────────────────────────────────────────
        scores = self._outer_scores(lvs, measurement, weights, X_std, ind_idx)
        converged = False
        n_iters   = 0
        max_delta = np.inf

        for iteration in range(self.max_iter):

            # (a) Inner approximation
            inner = self._inner_approx(
                lvs, scores, predecessors, successors, lv_idx
            )

            # (b) Mode A weight update: w_new_j = X_j^T · inner_j / n
            new_weights: Dict[str, np.ndarray] = {}
            for lv, inds in measurement.items():
                cols = [ind for ind in inds if ind in ind_idx]
                if not cols:
                    new_weights[lv] = weights[lv]
                    continue
                j       = lv_idx[lv]
                X_block = X_std[:, [ind_idx[ind] for ind in cols]]
                w_raw   = X_block.T @ inner[:, j] / n_obs

                # Normalise so Var(η_j) = 1
                var_eta = w_raw @ (X_block.T @ X_block / n_obs) @ w_raw
                denom   = np.sqrt(abs(var_eta))
                w_norm  = w_raw / max(denom, 1e-12)
                new_weights[lv] = w_norm

            # (c) Recompute scores
            new_scores = self._outer_scores(
                lvs, measurement, new_weights, X_std, ind_idx
            )

            # (d) Convergence: max absolute weight change across all LVs
            max_delta = max(
                np.max(np.abs(new_weights[lv] - weights[lv]))
                for lv in lvs
            )
            weights = new_weights
            scores  = new_scores
            n_iters = iteration + 1

            if max_delta < self.tolerance:
                converged = True
                break

        if not converged:
            warnings.append(
                f"PLS-SEM did not fully converge after {self.max_iter} iterations "
                f"(final Δw = {max_delta:.2e}). Results may be unreliable for "
                f"models with near-collinear constructs or very small samples."
            )

        # ── 5. Fix sign indeterminacy ──────────────────────────────────────────
        # Convention: each LV score correlates positively with a majority of its
        # indicators. Flip the score (and weights) if not.
        scores, weights = self._fix_signs(
            lvs, measurement, scores, X_std, ind_idx, weights
        )

        # ── 6. Outer loadings = cor(indicator, LV score) ──────────────────────
        outer_loadings  = self._compute_loadings(lvs, measurement, scores, X_std, ind_idx)
        outer_weights_d = self._weights_to_dict(lvs, measurement, weights, ind_idx)

        # ── 7. Structural OLS ──────────────────────────────────────────────────
        path_coefs, r_sq = self._structural_ols(lvs, structural, scores, lv_idx)

        # ── 8. SRMR ────────────────────────────────────────────────────────────
        srmr = self._compute_srmr(
            outer_loadings, measurement, all_indicators, df, scores, lvs, lv_idx
        )

        return PLSResult(
            scores            = pd.DataFrame(scores, columns=lvs),
            outer_weights     = outer_weights_d,
            outer_loadings    = outer_loadings,
            path_coefficients = path_coefs,
            r_squared         = r_sq,
            srmr              = srmr,
            n_iterations      = n_iters,
            converged         = converged,
            n_obs             = n_obs,
            warnings          = warnings,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _outer_scores(
        self,
        lvs:         List[str],
        measurement: Dict[str, List[str]],
        weights:     Dict[str, np.ndarray],
        X_std:       np.ndarray,
        ind_idx:     Dict[str, int],
    ) -> np.ndarray:
        """LV scores matrix (n × q) from outer weights."""
        n = X_std.shape[0]
        scores = np.zeros((n, len(lvs)))
        for j, lv in enumerate(lvs):
            cols = [ind for ind in measurement[lv] if ind in ind_idx]
            if not cols:
                continue
            X_block      = X_std[:, [ind_idx[ind] for ind in cols]]
            w            = weights[lv][: len(cols)]
            scores[:, j] = X_block @ w
        return scores

    def _inner_approx(
        self,
        lvs:          List[str],
        scores:       np.ndarray,
        predecessors: Dict[str, List[str]],
        successors:   Dict[str, List[str]],
        lv_idx:       Dict[str, int],
    ) -> np.ndarray:
        """
        Path weighting scheme inner approximation.

        For each LV j:
          Incoming edges (k → j): OLS regression coefficient β_jk
          Outgoing edges (j → k): Pearson correlation cor(η_j, η_k)
        Inner score: Σ_k e_jk · η_k
        """
        n, q = scores.shape
        inner = np.zeros_like(scores)

        if self.scheme == "centroid":
            # Simpler alternative: e_jk = sign(cor) for any connected pair
            for j in range(q):
                lv_j = lvs[j]
                connected = (
                    predecessors.get(lv_j, []) +
                    successors.get(lv_j, [])
                )
                for lv_k in connected:
                    k = lv_idx.get(lv_k)
                    if k is None:
                        continue
                    r = _safe_corr(scores[:, j], scores[:, k])
                    inner[:, j] += np.sign(r) * scores[:, k]
            return inner

        # Path weighting scheme (default)
        for j, lv_j in enumerate(lvs):
            preds = [p for p in predecessors.get(lv_j, []) if p in lv_idx]
            succs = [s for s in successors.get(lv_j,   []) if s in lv_idx]

            if not preds and not succs:
                # Isolated LV — inner score = outer score (no structural links)
                inner[:, j] = scores[:, j]
                continue

            # Incoming paths → OLS beta coefficients
            if preds:
                X_in = np.column_stack([scores[:, lv_idx[k]] for k in preds])
                X_in_c = np.column_stack([np.ones(n), X_in])
                try:
                    beta = np.linalg.lstsq(X_in_c, scores[:, j], rcond=None)[0][1:]
                    for idx_k, k in enumerate(preds):
                        inner[:, j] += beta[idx_k] * scores[:, lv_idx[k]]
                except Exception as _e:
                    # Fallback: use raw scores if OLS fails
                    for k in preds:
                        inner[:, j] += scores[:, lv_idx[k]]

            # Outgoing paths → correlation weight
            for lv_k in succs:
                k = lv_idx[lv_k]
                r = _safe_corr(scores[:, j], scores[:, k])
                inner[:, j] += r * scores[:, k]

        return inner

    def _fix_signs(
        self,
        lvs:         List[str],
        measurement: Dict[str, List[str]],
        scores:      np.ndarray,
        X_std:       np.ndarray,
        ind_idx:     Dict[str, int],
        weights:     Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Resolve sign indeterminacy.
        Flip LV score (and its weights) when it negatively correlates
        with a majority of its indicators.
        """
        for j, lv in enumerate(lvs):
            cols = [ind for ind in measurement[lv] if ind in ind_idx]
            if not cols:
                continue
            n_pos = sum(
                1 for ind in cols
                if _safe_corr(scores[:, j], X_std[:, ind_idx[ind]]) > 0
            )
            if n_pos < len(cols) / 2:
                scores[:, j]  = -scores[:, j]
                weights[lv]   = -weights[lv]
        return scores, weights

    def _compute_loadings(
        self,
        lvs:         List[str],
        measurement: Dict[str, List[str]],
        scores:      np.ndarray,
        X_std:       np.ndarray,
        ind_idx:     Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        """Outer loadings = cor(indicator, LV score)."""
        loadings: Dict[str, Dict[str, float]] = {}
        for j, lv in enumerate(lvs):
            loadings[lv] = {}
            for ind in measurement[lv]:
                if ind not in ind_idx:
                    continue
                r = _safe_corr(scores[:, j], X_std[:, ind_idx[ind]])
                loadings[lv][ind] = round(r, 8)
        return loadings

    def _weights_to_dict(
        self,
        lvs:         List[str],
        measurement: Dict[str, List[str]],
        weights:     Dict[str, np.ndarray],
        ind_idx:     Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        """Convert weight arrays → {lv: {indicator: weight}}."""
        result: Dict[str, Dict[str, float]] = {}
        for lv in lvs:
            cols = [ind for ind in measurement[lv] if ind in ind_idx]
            w    = weights[lv]
            result[lv] = {
                ind: round(float(w[k]), 8)
                for k, ind in enumerate(cols)
            }
        return result

    def _structural_ols(
        self,
        lvs:        List[str],
        structural: List[dict],
        scores:     np.ndarray,
        lv_idx:     Dict[str, int],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        OLS for the structural (inner) model.
        For each endogenous LV: regress its score on all predictor LV scores.
        Returns (path_coefficients, r_squared).
        """
        from collections import defaultdict

        preds_by_lhs: Dict[str, List[str]] = defaultdict(list)
        for rel in structural:
            preds_by_lhs[rel["lhs"]].append(rel["rhs"])

        path_coefs: Dict[str, Dict[str, float]] = {}
        r_squared:  Dict[str, float]             = {}
        n = scores.shape[0]

        for lhs, rhs_list in preds_by_lhs.items():
            j = lv_idx.get(lhs)
            if j is None:
                continue
            rhs_valid = [r for r in rhs_list if r in lv_idx]
            if not rhs_valid:
                continue

            y = scores[:, j]
            X = np.column_stack(
                [np.ones(n)] + [scores[:, lv_idx[r]] for r in rhs_valid]
            )
            try:
                beta  = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat = X @ beta
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2     = max(1.0 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0
                r_squared[lhs] = round(r2, 6)

                path_coefs.setdefault(lhs, {})
                for idx_r, rhs in enumerate(rhs_valid):
                    path_coefs[lhs][rhs] = round(float(beta[idx_r + 1]), 8)
            except Exception as _e:  # B112
                logger.debug("Non-critical exception suppressed: %s", _e)
                continue

        return path_coefs, r_squared

    def _compute_srmr(
        self,
        outer_loadings: Dict[str, Dict[str, float]],
        measurement:    Dict[str, List[str]],
        all_indicators: List[str],
        df:             pd.DataFrame,
        scores:         np.ndarray,
        lvs:            List[str],
        lv_idx:         Dict[str, int],
    ) -> Optional[float]:
        """
        SRMR for PLS-SEM via model-implied correlation matrix.

        Model-implied correlation matrix Σ (indicator-level):
          Diagonal: Σ_ii = 1  (we model correlations, not covariances)
          Within-block  (same LV):      Σ_ij = λ_i · λ_j
          Cross-block   (different LVs): Σ_ij = λ_i · φ_ab · λ_j
            where φ_ab = actual cor(η_a, η_b) from PLS scores

        SRMR formula (Henseler, Ringle & Sarstedt 2015):
          SRMR = sqrt( 2/(p(p+1)) · Σ_{i≥j} ( (s_ij − σ_ij) / sqrt(s_ii·s_jj) )² )
        """
        try:
            obs = [ind for ind in all_indicators if ind in df.columns]
            p   = len(obs)
            if p < 2:
                return None

            obs_idx  = {ind: i for i, ind in enumerate(obs)}

            # Which LV does each indicator belong to?
            lv_for_ind: Dict[str, str] = {}
            for lv, inds in measurement.items():
                for ind in inds:
                    lv_for_ind[ind] = lv

            # Loading lookup (indicator → outer loading)
            lam: Dict[str, float] = {}
            for lv, ind_map in outer_loadings.items():
                for ind, loading in ind_map.items():
                    lam[ind] = loading

            # Inter-construct correlation matrix from actual PLS scores
            q   = len(lvs)
            phi = np.eye(q)
            if q > 1:
                phi = np.corrcoef(scores.T)
                phi = np.clip(phi, -1.0, 1.0)  # guard floating-point drift

            # Sample correlation matrix
            S = df[obs].corr(numeric_only=True).values

            # Build model-implied correlation matrix Σ
            Sigma = np.zeros((p, p))
            for i, ind_i in enumerate(obs):
                lv_i  = lv_for_ind.get(ind_i)
                lam_i = lam.get(ind_i, 0.0)
                for j, ind_j in enumerate(obs):
                    lv_j  = lv_for_ind.get(ind_j)
                    lam_j = lam.get(ind_j, 0.0)

                    if i == j:
                        Sigma[i, j] = 1.0  # correlation matrix diagonal
                    elif lv_i is None or lv_j is None:
                        Sigma[i, j] = S[i, j]  # passthrough for unmapped vars
                    elif lv_i == lv_j:
                        Sigma[i, j] = lam_i * lam_j  # within-block: λ_i · λ_j
                    else:
                        li = lv_idx.get(lv_i, 0)
                        lj = lv_idx.get(lv_j, 0)
                        Sigma[i, j] = lam_i * phi[li, lj] * lam_j  # cross-block

            # SRMR (lower triangle including diagonal)
            total = 0.0
            count = 0
            for i in range(p):
                for j in range(i + 1):  # j ≤ i
                    denom = abs(S[i, i] * S[j, j]) ** 0.5
                    if denom > 0:
                        total += ((S[i, j] - Sigma[i, j]) / denom) ** 2
                        count += 1

            if count == 0:
                return None
            return round(float(np.sqrt(2.0 * total / (p * (p + 1)))), 6)

        except Exception as _e:
            return None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that never returns NaN (returns 0.0 on degenerate input)."""
    try:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        r = float(np.corrcoef(a, b)[0, 1])
        return r if not np.isnan(r) else 0.0
    except Exception as _e:
        return 0.0


def pls_loadings_to_list(
    outer_loadings: Dict[str, Dict[str, float]],
    measurement:    Dict[str, List[str]],
) -> Dict[str, List[float]]:
    """
    Convert {lv: {ind: λ}} → {lv: [λ, λ, ...]} in indicator order.
    Consumed by engine._compute_ave / _compute_composite_reliability.
    """
    result: Dict[str, List[float]] = {}
    for lv, inds in measurement.items():
        ind_map = outer_loadings.get(lv, {})
        result[lv] = [ind_map[ind] for ind in inds if ind in ind_map]
    return result
