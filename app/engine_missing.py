"""
engine_missing.py  —  NAVAL-SEM missing-data primitives
========================================================
Implements two feature groups:

  A1 — FIML log-likelihood  (Arbuckle 1996)
       fiml_loglik(df, model_params) → float
       Standalone evaluation of the FIML criterion at a given set of
       model-implied parameters.  The *estimation* path uses semopy's
       native obj='FIML'; this function is used for model-comparison
       diagnostics, unit tests, and AIC/BIC reporting on FIML fits.

  A2 — Deterministic and stochastic imputation
       regression_impute(df, target_col, predictor_cols) → pd.Series
       stochastic_regression_impute(df, target_col, predictor_cols, rng) → pd.Series
       bayesian_impute(df, target_col, predictor_cols, rng, n_draws=1) → list[pd.Series]

Literature
----------
Arbuckle, J. L. (1996). Full information estimation in the presence of
  incomplete data. In G. A. Marcoulides & R. E. Schumacker (Eds.),
  Advanced Structural Equation Modeling: Issues and Techniques (pp. 243-277).
  Erlbaum.

Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
  Wiley.

Enders, C. K. (2010). Applied Missing Data Analysis. Guilford Press.
  (Bayesian imputation framework in Ch. 9.)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("naval_sem.engine_missing")


# ══════════════════════════════════════════════════════════════════════════════
# A1 — Full Information Maximum Likelihood log-likelihood
# ══════════════════════════════════════════════════════════════════════════════

def fiml_loglik(
    df: pd.DataFrame,
    model_params: dict,
) -> float:
    """
    Evaluate the FIML log-likelihood at a given set of model-implied parameters.

    Partitions rows by their missing-data pattern (the set of observed column
    indices), then sums each pattern's contribution to the multivariate normal
    log-likelihood using only the observed sub-vector and the corresponding
    sub-matrix of the model-implied covariance (Arbuckle 1996, eq. 4–6).

    For a complete row, this reduces exactly to the ordinary ML criterion.

    Parameters
    ----------
    df : pd.DataFrame
        Observed data with ``float`` columns.  NaN encodes missing values.
        Columns may be a superset of ``model_params["variables"]``; only the
        modelled variables are used.
    model_params : dict
        Must contain:

        ``"variables"`` : list[str]
            Ordered variable names.  Defines the rows/columns of ``sigma``.
        ``"sigma"``     : array-like, shape (p, p)
            Model-implied covariance matrix Σ(θ).  Must be positive definite.

        Optional:

        ``"mu"``        : array-like, shape (p,)
            Model-implied mean vector μ(θ).  Defaults to the vector of column
            means computed from the complete cases in *df*.

    Returns
    -------
    float
        FIML log-likelihood ℓ(θ; data) = Σ_i ℓ_i.
        Returns ``-inf`` if any pattern sub-matrix is non-positive-definite.

    Raises
    ------
    KeyError
        If ``"variables"`` or ``"sigma"`` is missing from *model_params*.
    ValueError
        If ``sigma`` is not square or its dimension differs from
        ``len(variables)``.

    Notes
    -----
    Pattern-level log-likelihood contribution (Arbuckle 1996, p. 254):

        ℓ_pattern = −(n_p / 2) [k_p log(2π) + log |Σ_p|]
                    − (1/2) Σ_{i ∈ pattern} (x_i − μ_p)ᵀ Σ_p⁻¹ (x_i − μ_p)

    where *p* indexes the pattern, n_p is the number of rows in the pattern,
    k_p is the number of observed variables, Σ_p and μ_p are the
    corresponding sub-matrix and sub-vector of Σ and μ.
    """
    variables: list[str] = list(model_params["variables"])
    sigma_full = np.asarray(model_params["sigma"], dtype=float)
    p = len(variables)

    if sigma_full.ndim != 2 or sigma_full.shape != (p, p):
        raise ValueError(
            f"fiml_loglik: sigma must be ({p}, {p}), got {sigma_full.shape}."
        )

    # Default μ from complete-case column means
    if "mu" in model_params and model_params["mu"] is not None:
        mu_full = np.asarray(model_params["mu"], dtype=float)
        if mu_full.shape != (p,):
            raise ValueError(
                f"fiml_loglik: mu must be length {p}, got {mu_full.shape}."
            )
    else:
        col_means = df[variables].mean(skipna=True)
        mu_full = col_means.fillna(0.0).values.astype(float)

    # Map variable names → column indices in sigma / mu
    var_idx: dict[str, int] = {v: i for i, v in enumerate(variables)}

    # Restrict df to modelled variables; align column order to variables list
    avail_vars = [v for v in variables if v in df.columns]
    if not avail_vars:
        raise ValueError("fiml_loglik: none of model_params['variables'] found in df.")

    # Build data matrix (n × len(avail_vars)), NaN for missing
    data_np = df[avail_vars].values.astype(float)
    col_indices_in_sigma = [var_idx[v] for v in avail_vars]  # maps data col → sigma index
    n_rows = data_np.shape[0]

    # Group rows by observed-column pattern  ─────────────────────────────────
    # Key = tuple of *local* column indices (indices into avail_vars) that are finite.
    pattern_map: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n_rows):
        obs_local = tuple(int(j) for j in np.where(np.isfinite(data_np[i]))[0])
        pattern_map[obs_local].append(i)

    log_2pi = np.log(2.0 * np.pi)
    total_loglik = 0.0

    for obs_local_cols, row_indices in pattern_map.items():
        k = len(obs_local_cols)
        if k == 0:
            # All variables missing for this row — contribute 0
            continue

        # Sub-matrix of sigma and sub-vector of mu for this pattern
        full_cols = [col_indices_in_sigma[j] for j in obs_local_cols]
        sigma_p = sigma_full[np.ix_(full_cols, full_cols)]
        mu_p = mu_full[full_cols]

        # Invert and log-det of sub-sigma
        try:
            sign, logdet_p = np.linalg.slogdet(sigma_p)
            if sign <= 0:
                return float("-inf")
            sigma_p_inv = np.linalg.inv(sigma_p)
        except np.linalg.LinAlgError:
            return float("-inf")

        # Gather data for this pattern
        X_p = data_np[np.ix_(row_indices, list(obs_local_cols))]  # (n_p, k)
        n_p = len(row_indices)
        residuals = X_p - mu_p                                      # (n_p, k)

        # Σ_i (x_i − μ_p)ᵀ Σ_p⁻¹ (x_i − μ_p)  using einsum for speed
        mahal = float(np.einsum("ni,ij,nj->", residuals, sigma_p_inv, residuals))

        pattern_loglik = (
            -0.5 * n_p * (k * log_2pi + logdet_p)
            - 0.5 * mahal
        )
        total_loglik += pattern_loglik

    return float(total_loglik)


# ══════════════════════════════════════════════════════════════════════════════
# A2 — Imputation functions
# ══════════════════════════════════════════════════════════════════════════════

def _check_predictor_completeness(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
) -> None:
    """
    Raise ValueError if any row with a missing *target_col* also has a missing
    value in any *predictor_col*.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list[str]

    Raises
    ------
    ValueError
        If any row needing imputation has a missing predictor.
        Includes a count and the first few affected index values.
    """
    missing_target_mask = df[target_col].isna()
    if not missing_target_mask.any():
        return

    rows_to_impute = df.loc[missing_target_mask, predictor_cols]
    bad_mask = rows_to_impute.isna().any(axis=1)
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        bad_indices = rows_to_impute.index[bad_mask].tolist()
        raise ValueError(
            f"regression imputation: {bad_count} row(s) need imputation for "
            f"'{target_col}' but have missing predictor values "
            f"(first indices: {bad_indices[:5]}). "
            "Impute predictors first or reduce the predictor set."
        )


def _ols_fit(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit OLS on complete cases.

    Returns
    -------
    beta : np.ndarray, shape (1 + len(predictor_cols),)
        [intercept, coef_1, ..., coef_k]
    mse : float
        Mean squared error = SSE / max(n_complete − k − 1, 1)
    X_aug_complete : np.ndarray
        Design matrix for complete-case rows (with intercept column prepended).
    """
    complete = df.dropna(subset=[target_col] + predictor_cols)
    if len(complete) < 2:
        raise ValueError(
            f"Not enough complete cases to fit OLS for target='{target_col}'. "
            f"Found {len(complete)} complete row(s); need ≥ 2."
        )

    X = complete[predictor_cols].values.astype(float)
    y = complete[target_col].values.astype(float)
    X_aug = np.column_stack([np.ones(len(X)), X])

    beta, residuals, rank, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    y_hat = X_aug @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    dof = max(len(y) - len(beta), 1)
    mse = sse / dof

    return beta, mse, X_aug


def regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
) -> pd.Series:
    """
    Single imputation via OLS regression on complete cases.

    Fits OLS using rows where both *target_col* and all *predictor_cols* are
    observed, then predicts missing *target_col* values from observed predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  May contain NaN.
    target_col : str
        Column to impute.
    predictor_cols : list[str]
        Predictor columns.  Must be present in *df*.

    Returns
    -------
    pd.Series
        Copy of ``df[target_col]`` with NaN replaced by OLS-predicted values.
        Preserves the original dtype of the column.

    Raises
    ------
    ValueError
        If any row needing imputation has a missing predictor value.
        If fewer than 2 complete cases are available for fitting.

    Notes
    -----
    Point-predicted values give underestimated variance (variance
    shrinkage).  Use :func:`stochastic_regression_impute` when the
    imputed column will be used in variance estimation or as an outcome.
    """
    _check_predictor_completeness(df, target_col, predictor_cols)

    missing_mask = df[target_col].isna()
    if not missing_mask.any():
        return df[target_col].copy()

    beta, _, _ = _ols_fit(df, target_col, predictor_cols)

    X_miss = df.loc[missing_mask, predictor_cols].values.astype(float)
    X_miss_aug = np.column_stack([np.ones(len(X_miss)), X_miss])
    predictions = X_miss_aug @ beta

    result = df[target_col].copy()
    result.loc[missing_mask] = predictions
    return result


def stochastic_regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    rng: np.random.Generator,
) -> pd.Series:
    """
    Single stochastic regression imputation.

    Fits the same OLS model as :func:`regression_impute`, then adds
    independent Gaussian noise N(0, MSE) to each imputed value, where MSE
    is the residual mean square from the complete-case fit.  This preserves
    the variance of the imputed column (unlike plain regression imputation,
    which artificially deflates variability).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list[str]
    rng : np.random.Generator
        NumPy random generator for reproducibility (e.g.
        ``np.random.default_rng(42)``).

    Returns
    -------
    pd.Series
        Copy of ``df[target_col]`` with NaN replaced by stochastic predictions.

    Raises
    ------
    ValueError
        If any row needing imputation has a missing predictor value.
        If fewer than 2 complete cases are available for fitting.

    Notes
    -----
    Rubin (1987, p. 168) shows that stochastic regression imputation is
    proper for MCAR and MAR mechanisms when combined with multiple imputation.
    For a single imputation it still provides better variance estimates than
    plain regression imputation.
    """
    _check_predictor_completeness(df, target_col, predictor_cols)

    missing_mask = df[target_col].isna()
    if not missing_mask.any():
        return df[target_col].copy()

    beta, mse, _ = _ols_fit(df, target_col, predictor_cols)
    sigma_resid = np.sqrt(max(mse, 0.0))

    X_miss = df.loc[missing_mask, predictor_cols].values.astype(float)
    X_miss_aug = np.column_stack([np.ones(len(X_miss)), X_miss])
    predictions = X_miss_aug @ beta
    noise = rng.normal(loc=0.0, scale=sigma_resid, size=len(predictions))

    result = df[target_col].copy()
    result.loc[missing_mask] = predictions + noise
    return result


def bayesian_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    rng: np.random.Generator,
    n_draws: int = 1,
) -> list[pd.Series]:
    """
    Multiple imputation via Normal-Inverse-Gamma (NIG) posterior draws.

    Rather than fixing β at the OLS point estimate, draws β and σ² jointly
    from their NIG posterior given the complete-case data, then generates
    predicted values plus residual noise for each missing cell.  Calling with
    ``n_draws > 1`` produces *m* independent imputed datasets suitable for
    Rubin's (1987) combining rules.

    Posterior derivation (Enders 2010, Ch. 9)
    ------------------------------------------
    Diffuse (improper) prior on (β, σ²):
        p(β, σ²) ∝ 1/σ²

    With *n* complete observations and design matrix X_c (including intercept):

        V_n   = (X_c^T X_c)^{−1}                  [posterior covariance scaling]
        β_n   = (X_c^T X_c)^{−1} X_c^T y_c        [OLS estimate = posterior mode]
        a_n   = n / 2
        b_n   = SSE / 2  where SSE = ||y_c − X_c β_n||²

    Each draw:
        1.  σ²* ~ InvGamma(a_n, b_n)  →  1/Gamma(a_n, 1/b_n)
        2.  β*  ~ MVN(β_n, σ²* V_n)
        3.  ŷ_i* = X_miss β*  +  ε_i*,  ε_i* ~ N(0, σ²*)

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list[str]
    rng : np.random.Generator
    n_draws : int, optional
        Number of imputed datasets to return.  Default 1.

    Returns
    -------
    list of pd.Series, length ``n_draws``
        Each element is a copy of ``df[target_col]`` with NaN replaced by
        imputed values from one posterior draw.

    Raises
    ------
    ValueError
        If any row needing imputation has a missing predictor value.
        If fewer than 2 complete cases are available for fitting.
        If ``n_draws < 1``.

    Notes
    -----
    The NIG posterior with an improper flat prior reduces exactly to the
    data-informed posterior and produces draws that are broader than a
    simple bootstrap would suggest — this is the source of proper MI
    coverage (Rubin 1987, Theorem 3.1).
    """
    if n_draws < 1:
        raise ValueError(f"bayesian_impute: n_draws must be >= 1, got {n_draws}.")

    _check_predictor_completeness(df, target_col, predictor_cols)

    missing_mask = df[target_col].isna()
    if not missing_mask.any():
        return [df[target_col].copy() for _ in range(n_draws)]

    # ── Complete-case fit ─────────────────────────────────────────────────────
    complete = df.dropna(subset=[target_col] + predictor_cols)
    if len(complete) < 2:
        raise ValueError(
            f"Not enough complete cases for bayesian_impute on '{target_col}': "
            f"found {len(complete)}, need ≥ 2."
        )

    X_c = np.column_stack(
        [np.ones(len(complete)), complete[predictor_cols].values.astype(float)]
    )
    y_c = complete[target_col].values.astype(float)
    n_comp = len(y_c)
    k = X_c.shape[1]  # intercept + n predictors

    # OLS: β_n = (X'X)^{-1} X'y
    XtX = X_c.T @ X_c
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)  # singular guard for near-collinear predictors
    beta_n = XtX_inv @ (X_c.T @ y_c)

    sse = float(np.sum((y_c - X_c @ beta_n) ** 2))
    a_n = n_comp / 2.0
    b_n = sse / 2.0

    # Design matrix for rows to impute
    X_miss = df.loc[missing_mask, predictor_cols].values.astype(float)
    X_miss_aug = np.column_stack([np.ones(len(X_miss)), X_miss])
    n_miss = len(X_miss_aug)

    results: list[pd.Series] = []

    for _ in range(n_draws):
        # Step 1: draw σ²* ~ InvGamma(a_n, b_n)
        # IG(a, b): if g ~ Gamma(a, 1/b) then 1/g ~ IG(a, b)
        if b_n < 1e-12:
            # Near-perfect fit: variance is essentially zero — use small floor
            sigma2_star = 1e-10
        else:
            g = rng.gamma(shape=a_n, scale=1.0 / b_n)
            sigma2_star = 1.0 / max(g, 1e-300)

        # Step 2: draw β* ~ MVN(β_n, σ²* (X'X)^{-1})
        cov_beta = sigma2_star * XtX_inv
        # Ensure PD via Cholesky (numerical jitter when needed)
        try:
            L = np.linalg.cholesky(cov_beta)
            z = rng.standard_normal(k)
            beta_star = beta_n + L @ z
        except np.linalg.LinAlgError:
            # Jitter diagonal to restore PD
            jitter = 1e-8 * np.eye(k)
            try:
                L = np.linalg.cholesky(cov_beta + jitter)
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.maximum(np.diag(cov_beta), 0.0)))
            z = rng.standard_normal(k)
            beta_star = beta_n + L @ z

        # Step 3: predict + add residual noise
        y_pred = X_miss_aug @ beta_star
        noise = rng.normal(loc=0.0, scale=np.sqrt(sigma2_star), size=n_miss)
        imputed_vals = y_pred + noise

        imputed_series = df[target_col].copy()
        imputed_series.loc[missing_mask] = imputed_vals
        results.append(imputed_series)

    return results
