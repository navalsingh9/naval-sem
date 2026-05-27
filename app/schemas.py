"""
Pydantic schemas for all API responses.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PathParameter(BaseModel):
    lhs: str
    op: str            # =~ (measurement) or ~ (structural) or ~~ (covariance)
    rhs: str
    estimate: float
    std_estimate: Optional[float] = None   # standardized estimate (std.all)
    std_error: Optional[float] = None      # None for ~~ rows — not hypothesis-tested
    z_value: Optional[float] = None        # None for ~~ rows — not hypothesis-tested
    p_value: Optional[float] = None        # None for ~~ rows — not hypothesis-tested
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    significant: bool = False              # always False when p_value is None


class FitIndices(BaseModel):
    cfi: Optional[float] = None
    tli: Optional[float] = None    # Tucker-Lewis Index (NNFI)
    rmsea: Optional[float] = None
    rmsea_ci_lower: Optional[float] = None
    rmsea_ci_upper: Optional[float] = None
    srmr: Optional[float] = None
    chi_square: Optional[float] = None
    df: Optional[int] = None
    p_value: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    r_squared: Optional[Dict[str, float]] = None

    # Measurement validity metrics (v0.3)
    ave: Optional[Dict[str, float]] = None
    composite_reliability: Optional[Dict[str, float]] = None
    cronbach_alpha: Optional[Dict[str, float]] = None
    fornell_larcker: Optional[Dict[str, Dict[str, float]]] = None
    fornell_larcker_pass: Optional[bool] = None

    # Fit verdict helpers
    cfi_acceptable: Optional[bool] = None    # CFI >= 0.90
    cfi_good: Optional[bool] = None          # CFI >= 0.95
    tli_acceptable: Optional[bool] = None    # TLI >= 0.90
    tli_good: Optional[bool] = None          # TLI >= 0.95
    rmsea_acceptable: Optional[bool] = None  # RMSEA <= 0.08
    rmsea_good: Optional[bool] = None        # RMSEA <= 0.06
    srmr_good: Optional[bool] = None         # SRMR <= 0.08


class BootstrapResult(BaseModel):
    n_samples: int
    parameters: List[Dict[str, Any]]
    converged_pct: float


class HTMTEntry(BaseModel):
    construct_a: str
    construct_b: str
    htmt: float
    acceptable: bool    # HTMT < 0.90


class HTMTResult(BaseModel):
    matrix: List[HTMTEntry]
    all_acceptable: bool


# ── v0.4 schemas ──────────────────────────────────────────────────────────────

class OuterWeightEntry(BaseModel):
    lv: str
    latent_variable: Optional[str] = None  # alias for lv — JSON compat with frontend
    indicator: str
    estimate: float          # point estimate from full-data fit
    bs_mean: float           # mean across bootstrap samples
    bs_se: float             # bootstrap standard error
    ci_lower_95: float
    ci_upper_95: float
    t_stat: Optional[float] = None   # estimate / bs_se
    significant: bool                # CI excludes zero

    def model_post_init(self, __context: Any) -> None:
        """Keep latent_variable in sync with lv so the JSON always has both."""
        if self.latent_variable is None:
            object.__setattr__(self, "latent_variable", self.lv)


class VIFEntry(BaseModel):
    lv: str
    indicator: str
    vif: float
    acceptable: bool    # VIF < 5.0 (common threshold); < 3.3 for PLS-SEM strict


class F2Entry(BaseModel):
    lhs: str            # dependent variable
    rhs: str            # predictor being tested
    r2_full: float      # R² with predictor included
    r2_reduced: float   # R² with predictor removed
    f2: float           # Cohen's f² = (R²_full - R²_reduced) / (1 - R²_full)
    effect: str         # "negligible" | "small" | "medium" | "large"


class IndirectEffect(BaseModel):
    from_var: str
    to_var: str
    through: List[str]              # mediator variable(s) in order
    indirect_effect: float
    bs_se: Optional[float] = None
    ci_lower_95: Optional[float] = None
    ci_upper_95: Optional[float] = None
    significant: Optional[bool] = None   # True when CI excludes zero


class IndirectResult(BaseModel):
    effects: List[IndirectEffect]
    total_effects: Dict[str, Dict[str, float]]  # {from_var: {to_var: total}}


# ── v0.7 summary schemas ──────────────────────────────────────────────────────────────────

class StructuralPathSummary(BaseModel):
    """One row per structural path in the inner model."""
    from_var:    str
    to_var:      str
    beta:        float
    t_stat:      Optional[float] = None
    p_value:     Optional[float] = None
    ci_lower_95: Optional[float] = None
    ci_upper_95: Optional[float] = None
    significant: bool
    f2:          Optional[float] = None
    f2_label:    Optional[str]  = None


class ConstructValiditySummary(BaseModel):
    """One row per latent variable."""
    construct_name:        str
    n_indicators:          int
    avg_loading:           Optional[float] = None
    min_loading:           Optional[float] = None
    ave:                   Optional[float] = None
    ave_sqrt:              Optional[float] = None
    composite_reliability: Optional[float] = None
    cronbach_alpha:        Optional[float] = None
    ave_ok:                Optional[bool]  = None
    cr_ok:                 Optional[bool]  = None
    alpha_ok:              Optional[bool]  = None


class ModelSummary(BaseModel):
    """High-level digest of ModelResult for the Results Summary panel."""
    algorithm:            str
    n_obs:                int
    bootstrap_n:          int
    structural_paths:     List[StructuralPathSummary]
    construct_validity:   List[ConstructValiditySummary]
    fornell_larcker_pass: Optional[bool]  = None
    all_loadings_ok:      Optional[bool]  = None
    srmr:                 Optional[float] = None
    srmr_ok:              Optional[bool]  = None
    r_squared:            Optional[Dict[str, float]] = None
    cfi:                  Optional[float] = None
    rmsea:                Optional[float] = None
    verdict:              str


class ModelResult(BaseModel):
    algorithm: str
    n_obs: int
    n_params: int
    converged: bool
    parameters: List[PathParameter]
    fit: FitIndices
    latent_variables: List[str]
    observed_variables: List[str]
    bootstrap: Optional[BootstrapResult] = None
    bootstrap_error: Optional[str] = None
    # v0.4
    vif: Optional[List[VIFEntry]] = None
    f2: Optional[List[F2Entry]] = None
    indirect: Optional[IndirectResult] = None
    outer_weights: Optional[List[OuterWeightEntry]] = None
    warnings: List[str] = []
    # v0.6 — reproducibility
    run_id: Optional[str] = None
    fingerprint: Optional[str] = None
    # v0.7 — results summary
    summary: Optional[ModelSummary] = None
    # v0.6 — higher-order constructs
    hoc_type: Optional["HOCType"] = None
# ── v0.5 schemas ──────────────────────────────────────────────────────────────

class Q2Entry(BaseModel):
    lv: str                      # endogenous LV
    q2: float                    # Stone-Geisser Q² (1 - SSE/SSO)
    sse: float                   # sum of squared errors (blindfolded)
    sso: float                   # sum of squared observations
    omission_distance: int       # D used in the omission loop
    predictive_relevance: str    # "none" | "small" | "medium" | "large"


class PLSPredictEntry(BaseModel):
    lv: str                      # endogenous LV
    indicator: str
    rmse_model: float            # RMSE of PLS-SEM predictions
    rmse_lm: float               # RMSE of LM (naive) benchmark
    mae_model: float
    mae_lm: float
    q2_predict: float            # 1 - (RMSE_model² / RMSE_lm²)
    better_than_lm: bool         # model outperforms linear baseline


class CVPATResult(BaseModel):
    lv: str
    cvpat_statistic: float       # mean loss difference: LM - model
    p_value: Optional[float]     # one-sample t-test p-value
    significant: bool            # model significantly better than LM
    n_folds: int


class CMBMarkerResult(BaseModel):
    marker_variable: str
    correlations_with_substantive: Dict[str, float]   # {indicator: r}
    mean_marker_correlation: float
    max_marker_correlation: float
    cmb_concern: bool            # True when max r > 0.20 (Lindell & Whitney)
    note: str


class PredictResult(BaseModel):
    q2: List[Q2Entry]
    plspredict: Optional[List[PLSPredictEntry]] = None
    cvpat: Optional[List[CVPATResult]] = None



# ── v0.6 schemas ──────────────────────────────────────────────────────────────

class HOCType(str, Enum):
    """How a higher-order construct was estimated."""
    none = "none"
    repeated_indicator = "repeated_indicator"
    two_stage = "two_stage"


# ─── MICOM ────────────────────────────────────────────────────────────────────

class MICOMStep2Entry(BaseModel):
    """
    Step 2 — Compositional invariance for one construct.

    c = cor(X_g1 @ w_g1, X_g1 @ w_g2)  (cross-weighted, within group-1 data)
    Invariant when c ≥ ci_lower_95  (one-sided; c should be near 1.0).
    """
    lv_name: str
    correlation: float           # observed cross-weighted correlation
    ci_lower_95: float           # 5th percentile of permuted distribution
    invariant: bool              # correlation >= ci_lower_95


class MICOMStep3MeanEntry(BaseModel):
    """Step 3a — Composite mean equality for one construct."""
    lv_name: str
    mean_g1: float
    mean_g2: float
    mean_diff: float             # mean_g1 - mean_g2
    ci_lower_95: float           # 2.5th percentile of permuted distribution
    ci_upper_95: float           # 97.5th percentile
    invariant: bool              # 0.0 falls within [ci_lower_95, ci_upper_95]


class MICOMStep3VarEntry(BaseModel):
    """Step 3b — Composite variance equality for one construct."""
    lv_name: str
    var_g1: float
    var_g2: float
    var_ratio: float             # var_g1 / var_g2  (1.0 = equal)
    ci_lower_95: float
    ci_upper_95: float
    invariant: bool              # 1.0 falls within [ci_lower_95, ci_upper_95]


class MICOMResult(BaseModel):
    """
    Full MICOM output (Henseler, Ringle & Sarstedt 2016).

    partial_invariance (step 2 pass) is the minimum requirement for valid
    PLS-MGA path-coefficient comparisons.
    full_invariance requires step 2 + step 3a + step 3b.
    """
    n_permutations: int
    groups: List[str]
    step2: List[MICOMStep2Entry]
    step3_mean: List[MICOMStep3MeanEntry]
    step3_var: List[MICOMStep3VarEntry]
    full_invariance: bool
    partial_invariance: bool     # step 2 all-pass — sufficient for MGA path comparison


# ─── MGA ──────────────────────────────────────────────────────────────────────

class MGAGroupResult(BaseModel):
    """Per-group model fit — lightweight (no bootstrap / VIF / indirect)."""
    group_name: str              # stringified value of the grouping variable
    n_obs: int
    parameters: List[PathParameter]
    fit: FitIndices
    r_squared: Optional[Dict[str, float]] = None


class MGAPathDiff(BaseModel):
    """
    Bootstrap path-coefficient difference for one structural path, one group pair.

    Significant when the 95 % percentile CI excludes 0.
    """
    lhs: str
    rhs: str
    group_a: str
    group_b: str
    beta_a: float
    beta_b: float
    diff: float                  # beta_a − beta_b  (point estimate on full data)
    ci_lower_95: float           # 2.5th percentile of bootstrap distribution
    ci_upper_95: float           # 97.5th percentile
    significant: bool            # CI excludes 0


class MGAResult(BaseModel):
    """
    Multi-Group Analysis result.

    Includes per-group fit, pairwise bootstrap path-difference CIs,
    and (for 2-group analyses) MICOM measurement-invariance results.
    """
    grouping_variable: str
    groups: List[str]
    bootstrap_n: int
    group_results: List[MGAGroupResult]
    path_differences: List[MGAPathDiff]
    micom: Optional[MICOMResult] = None
    warnings: List[str] = []

ModelResult.model_rebuild()