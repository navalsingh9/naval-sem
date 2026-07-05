"""
scale.py — Scale development utilities for NAVAL-SEM v0.9.

CVI: compute_cvi()  |  EFA: compute_efa() (Session 3)
"""
import math
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import chi2 as chi2_dist
from sklearn.decomposition import FactorAnalysis

from app.schemas import CVIResult, ScaleDevelopmentResult


def compute_cvi(
    ratings_df: pd.DataFrame,
    n_experts: int,
    relevant_scale: tuple = (3, 4),
) -> CVIResult:
    """
    Compute Content Validity Index from an expert ratings matrix.

    ratings_df: rows = experts, columns = items, values = 1–4 Likert ratings.
    n_experts: number of experts (rows) the I-CVI proportions are computed over.
        Must equal ratings_df.shape[0] — see validation below.
    relevant_scale: rating values counted as "relevant" (default top two points
        of a 4-point relevance scale: 3 = quite relevant, 4 = highly relevant).
    """
    n_rows = ratings_df.shape[0]
    if n_experts != n_rows:
        raise ValueError(
            f"Number of experts ({n_experts}) does not match the number of "
            f"rows in the uploaded ratings file ({n_rows}). Each row must be "
            "one expert's ratings — every I-CVI proportion is computed as "
            "(experts rating the item as relevant) / n_experts, so a "
            "mismatch here silently produces invalid values (e.g. an I-CVI "
            "above 1.0). Fix the 'Number of experts' field or check that "
            "the file has the expected number of rows."
        )

    n_items = ratings_df.shape[1]

    # 1. I-CVI per item: proportion of experts rating the item as relevant.
    is_relevant = ratings_df.isin(relevant_scale)
    item_cvi_series = is_relevant.sum(axis=0) / n_experts
    item_cvi = {
        str(col): round(float(val), 4) for col, val in item_cvi_series.items()
    }

    # 2. S-CVI/Ave: mean of all I-CVI values.
    s_cvi_ave = round(float(item_cvi_series.mean()), 4)

    # 3. S-CVI/UA: proportion of items with I-CVI == 1.00.
    s_cvi_ua = round(float((item_cvi_series == 1.0).sum() / n_items), 4)

    # 4. Modified kappa (kappa*) per item, then averaged across items.
    k = math.ceil(n_experts / 2)
    p_c = math.comb(n_experts, k) * (0.5 ** n_experts)
    denom = 1 - p_c
    if denom == 0:
        denom = 1e-12
    kappa_per_item = (item_cvi_series - p_c) / denom
    kappa_star = round(float(kappa_per_item.mean()), 4)

    # 5. Interpretation.
    if s_cvi_ave >= 0.90 and bool((item_cvi_series >= 0.78).all()):
        interpretation = "Excellent"
    elif s_cvi_ave >= 0.80:
        interpretation = "Acceptable"
    else:
        interpretation = "Poor"

    # 6. Assemble result.
    return CVIResult(
        item_cvi=item_cvi,
        s_cvi_ave=s_cvi_ave,
        s_cvi_ua=s_cvi_ua,
        kappa_star=kappa_star,
        n_experts=n_experts,
        n_items=n_items,
        interpretation=interpretation,
    )


def compute_efa(
    df: pd.DataFrame,
    n_factors: Optional[int] = None,
    rotation: str = "varimax",
    log_fn=None,
) -> ScaleDevelopmentResult:
    """
    Exploratory Factor Analysis pipeline for scale development.

    Runs KMO and Bartlett's test of sphericity as suitability diagnostics,
    extracts factors via sklearn's FactorAnalysis with the requested
    rotation, and flags weak-loading and cross-loading items.

    df: rows = respondents, columns = items (numeric).
    n_factors: number of factors to extract; if None, uses the Kaiser
        criterion (eigenvalues > 1) on the correlation matrix.
    rotation: passed straight to sklearn's FactorAnalysis (e.g. "varimax").
    log_fn: optional callable(level: str, message: str) for progress logging.
    """

    def _log(message: str) -> None:
        if log_fn is not None:
            log_fn("info", message)

    items = list(df.columns)
    n_items = len(items)
    n_obs = df.shape[0]
    warnings: list = []

    # 1. KMO statistic (manual).
    _log("Computing correlation matrix and KMO statistic.")
    R = df.corr().values
    R_inv = np.linalg.pinv(R)
    diag_outer = np.sqrt(np.outer(np.diag(R_inv), np.diag(R_inv)))
    with np.errstate(divide="ignore", invalid="ignore"):
        U = -R_inv / diag_outer
    np.fill_diagonal(U, 0.0)

    off_diag_mask = ~np.eye(n_items, dtype=bool)
    r_sq_sum = float(np.sum(R[off_diag_mask] ** 2))
    u_sq_sum = float(np.sum(U[off_diag_mask] ** 2))
    kmo = round(r_sq_sum / (r_sq_sum + u_sq_sum), 4) if (r_sq_sum + u_sq_sum) > 0 else 0.0

    # 2. Bartlett's test of sphericity (H0: correlation matrix == identity).
    # Computed directly via the classical closed-form statistic rather than
    # pingouin.sphericity(), which implements Mauchly's test for
    # repeated-measures designs — a different null hypothesis.
    _log("Running Bartlett's test of sphericity.")
    sign, logdet = np.linalg.slogdet(R)
    bartlett_df = n_items * (n_items - 1) / 2
    if sign > 0:
        bartlett_chi2 = float(-(n_obs - 1 - (2 * n_items + 5) / 6) * logdet)
        bartlett_p = float(chi2_dist.sf(bartlett_chi2, bartlett_df))
    else:
        bartlett_chi2 = float("nan")
        bartlett_p = 1.0
    bartlett_chi2 = round(bartlett_chi2, 4)
    bartlett_p = round(bartlett_p, 4)

    # 3. Eigenvalues and Kaiser criterion.
    eigenvalues, _ = np.linalg.eigh(R)
    eigenvalues = sorted(eigenvalues, reverse=True)
    if n_factors is None:
        n_factors = int(sum(e > 1 for e in eigenvalues))
    n_factors = max(1, n_factors)
    _log(f"Retaining {n_factors} factor(s).")

    # 4. Factor extraction.
    # sklearn.FactorAnalysis uses the ML common-factor model — not PCA.
    # PCA is a different dimensionality-reduction technique with no latent factor assumption.
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation, max_iter=1000)
    fa.fit(df)
    L = fa.components_.T   # shape (n_items, n_factors)

    # 5. Loadings list.
    loadings = [
        {
            "item": col,
            "factor": int(np.argmax(np.abs(L[i, :])) + 1),
            "loading": round(float(L[i, np.argmax(np.abs(L[i, :]))]), 4),
        }
        for i, col in enumerate(items)
    ]

    # 6. Cross-loadings: second-highest |loading| > 0.30.
    cross_loadings = []
    if n_factors > 1:
        for i, col in enumerate(items):
            abs_loadings = np.abs(L[i, :])
            order = np.argsort(abs_loadings)[::-1]
            primary_idx, secondary_idx = order[0], order[1]
            secondary_val = abs_loadings[secondary_idx]
            if secondary_val > 0.30:
                cross_loadings.append({
                    "item": col,
                    "primary_factor": int(primary_idx + 1),
                    "secondary_factor": int(secondary_idx + 1),
                    "secondary_loading": round(float(L[i, secondary_idx]), 4),
                })
                warnings.append(f"Possible cross-loading: {col}")

    # 7. Variance explained per factor + cumulative.
    variance_explained = [
        round(float(np.sum(L[:, j] ** 2) / n_items), 4) for j in range(n_factors)
    ]
    cumulative_variance = round(float(sum(variance_explained)), 4)

    # 8. Remaining warnings.
    for i, col in enumerate(items):
        if np.max(np.abs(L[i, :])) < 0.40:
            warnings.append(f"Weak loading: {col}")
    if kmo < 0.60:
        warnings.append("KMO below acceptable threshold (0.60)")
    if bartlett_p > 0.05:
        warnings.append("Bartlett test not significant — EFA may not be appropriate")

    _log(
        f"EFA complete: KMO={kmo}, n_factors={n_factors}, "
        f"cumulative_variance={cumulative_variance}"
    )

    return ScaleDevelopmentResult(
        method=f"FA_{rotation.title()}",
        n_factors=n_factors,
        kmo=kmo,
        bartlett_chi2=bartlett_chi2,
        bartlett_p=bartlett_p,
        eigenvalues=[round(float(e), 4) for e in eigenvalues],
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        loadings=loadings,
        cross_loadings=cross_loadings,
        warnings=warnings,
    )