"""
Pydantic schemas for all API responses.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


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

    # v1.1 — Expanded fit indices (A6). TLI already existed above; the rest
    # are computed post-fit from chi2 / df / baseline chi2 / n, the same
    # inputs CFI and RMSEA already use (see engine._compute_expanded_fit_indices).
    gfi: Optional[float] = None          # Goodness of Fit Index (Jöreskog & Sörbom)
    agfi: Optional[float] = None         # Adjusted GFI (penalized for df)
    nfi: Optional[float] = None          # Normed Fit Index (Bentler-Bonett, 1980)
    hoelter_05: Optional[int] = None     # Hoelter's critical N, alpha = .05
    hoelter_01: Optional[int] = None     # Hoelter's critical N, alpha = .01
    ecvi: Optional[float] = None         # Expected Cross-Validation Index (Browne & Cudeck, 1989)
    pclose: Optional[float] = None       # p-value for H0: RMSEA <= .05 (test of close fit)

    # v1.1 — Plain-English fit verdict (A7). Built from the same thresholds
    # as the *_acceptable / *_good flags above — see engine._fit_verdict,
    # which is the single source of truth for both the booleans and this text.
    fit_verdict: Optional[str] = None


class BootstrapParameter(BaseModel):
    lhs: str
    op: str
    rhs: str
    estimate: float
    bs_mean: float
    bs_se: float
    ci_lower_95: float
    ci_upper_95: float
    significant: bool


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA FREEZE — v1.0
# Public result models below this line are stable. No fields may be renamed,
# removed, or have their types narrowed in patch or minor releases.
# New optional fields may be added (default=None). Breaking changes require v2.
# ═══════════════════════════════════════════════════════════════════════════════


class BootstrapResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_samples: int
    parameters: List[BootstrapParameter]
    converged_pct: float
    # v1.1 (A17) — NOTE: the feature ticket asked for annotations "per
    # indirect effect" on this class, but BootstrapResult.parameters holds
    # bootstrapped *direct*-path estimates (BootstrapParameter: lhs/op/rhs),
    # not indirect/mediated effects — those live in IndirectResult.effects
    # (see IndirectResult.annotations below, which is almost certainly the
    # field that request meant). Adding it here too, in case per-parameter
    # bootstrap sentences are also wanted for this list — each would use
    # engine_utils.annotate_indirect_effect(estimate, ci_lower_95, ci_upper_95),
    # whose CI-excludes-zero logic applies equally to a bootstrapped direct
    # path. Confirm which is intended before wiring engine.py; harmless
    # (empty list) either way until then.
    annotations: List[str] = []


class HTMTEntry(BaseModel):
    construct_a: str
    construct_b: str
    htmt: float
    acceptable: bool    # HTMT < 0.90
    ci_lower_95: Optional[float] = None
    ci_upper_95: Optional[float] = None
    ci_significant: Optional[bool] = None   # True when 95% CI upper bound < 0.90


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
    # v1.1 — plain-English indirect-effect annotations (A17). One sentence
    # per entry in `effects`, same order, via engine_utils.annotate_
    # indirect_effect(indirect_effect, ci_lower_95, ci_upper_95). See the
    # note on BootstrapResult.annotations above — this is likely the field
    # the ticket's "BootstrapResult (per indirect effect)" line meant.
    annotations: List[str] = []


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


class CVIResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_cvi: Dict[str, float]       # I-CVI per item (proportion rating >= 3)
    s_cvi_ave: float                 # mean of all I-CVIs
    s_cvi_ua: float                  # proportion of items with I-CVI = 1.00
    kappa_star: float                # mean modified kappa across items
    n_experts: int
    n_items: int
    interpretation: str              # "Excellent" / "Acceptable" / "Poor"


class ScaleDevelopmentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: str                      # e.g. "PCA_Varimax"
    n_factors: int
    kmo: float
    bartlett_chi2: float
    bartlett_p: float
    eigenvalues: List[float]
    variance_explained: List[float]  # per factor
    cumulative_variance: float
    loadings: List[Dict[str, Any]]   # [{item, factor, loading}]
    cross_loadings: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = []


class NomologicalResult(BaseModel):
    # `construct` shadows pydantic's BaseModel.construct() (the deprecated
    # v1-compat fast-instantiation classmethod), which pydantic warns about
    # at class-definition time. The Python attribute is renamed below;
    # alias= keeps the wire format identical — FastAPI serializes by alias
    # by default, so the JSON key is still "construct" and no consumer
    # (tests, frontend) needs to change.
    model_config = ConfigDict(populate_by_name=True)

    construct_name: str = Field(alias="construct")
    r_squared: float
    benchmark: float
    passed: bool
    interpretation: str              # "Substantial" / "Moderate" / "Weak" / "Not supported"


class MeasurementInvarianceLevel(BaseModel):
    model: str                       # "configural" | "metric" | "scalar"
    cfi: Optional[float] = None
    rmsea: Optional[float] = None
    srmr: Optional[float] = None
    delta_cfi: Optional[float] = None
    delta_rmsea: Optional[float] = None
    passed: bool


class MeasurementInvarianceResult(BaseModel):
    group_col: str
    groups: List[str]
    configural: MeasurementInvarianceLevel
    metric: MeasurementInvarianceLevel
    scalar: MeasurementInvarianceLevel
    partial_invariance: Optional[List[str]] = None   # items released
    conclusion: str   # "Full scalar" / "Partial scalar" / "Metric only" / "Configural only"


class ModelResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

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
    # v1.1 — missing data & normality (A1 + B1)
    missing_data_method: Optional[str] = None  # "listwise" | "FIML" | "mean"
    normality_check: Optional[Dict[str, Any]] = None  # Mardia (1970) test result
    # v1.1 — plain-English path-coefficient annotations (A17). One sentence
    # per regression-weight parameter (op == "~") in `parameters`, in the
    # same relative order — annotations[i] describes the i-th "~" row, not
    # necessarily the i-th row of `parameters` as a whole (which also holds
    # "=~" and "~~" rows). Populated via
    # engine_utils.annotate_path_coefficient(estimate, std_error, p_value).
    annotations: List[str] = []
# ── v0.5 schemas ──────────────────────────────────────────────────────────────

# ── v1.1 schemas — Imputation (A3) ────────────────────────────────────────────

class ImputationVariableStats(BaseModel):
    """Per-variable diagnostics returned by /impute."""
    pct_missing: float        # percentage of values that were missing before imputation
    n_missing: int            # absolute count of imputed cells
    method: str               # method actually applied ("regression" | "stochastic" | "bayesian")


class ImputationResult(BaseModel):
    """
    Diagnostics returned alongside imputed dataset(s) by POST /impute.

    ``between_imputation_variance`` is populated only for multiple-imputation
    runs (``m > 1``) and holds Rubin's (1987) between-imputation variance
    component  B = (1/(m−1)) Σ_j (q̄_j − q̄)²  for each imputed column,
    where q̄_j is the column mean of imputed values in draw *j*.
    """
    model_config = ConfigDict(extra="ignore")

    method: str                                          # "regression" | "stochastic" | "bayesian"
    n_imputed: int                                       # total number of cells imputed across all target_cols
    m: int = 1                                           # number of imputed datasets
    per_variable: Dict[str, ImputationVariableStats]     # one entry per target_col
    between_imputation_variance: Optional[Dict[str, float]] = None  # Rubin's B per column


class ImputeResponse(BaseModel):
    """
    Full response from POST /impute.

    ``imputed_datasets`` holds *m* datasets as lists of records (same format as
    ``df.to_dict(orient='records')``).  For ``m = 1`` the list has exactly one
    element.  Consumers should apply Rubin's rules across all *m* datasets
    before analysing results.
    """
    model_config = ConfigDict(extra="ignore")

    result: ImputationResult
    imputed_datasets: List[List[Dict[str, Any]]]        # m datasets, each as list-of-records




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
    rmse_lm: float                # RMSE of the LM (linear-regression) benchmark
    mae_model: float
    mae_lm: float
    q2_predict: float            # 1 - (RMSE_model² / RMSE_lm²)
    better_than_lm: bool         # model outperforms linear baseline

    # v1.1 (S4) — naive benchmark: every held-out value in a fold is
    # predicted using that fold's training-set mean for the indicator
    # (no predictors, no model at all). This is distinct from rmse_lm/
    # mae_lm above despite the older comment that used to call the LM
    # benchmark "naive" — rmse_lm is a real OLS regression; this is not.
    # Optional only for backward-compatible schema evolution (same
    # pattern as MGAPathDiff.p_value_henseler, v1.1 S1); always populated
    # once PLSpredict runs.
    rmse_naive: Optional[float] = None
    mae_naive: Optional[float] = None
    better_than_naive: Optional[bool] = None   # model outperforms naive baseline

    # v1.1 (S5) — per-indicator Shmueli et al. (2019) decision-rule
    # verdict (RMSE-based, matching better_than_lm above), extended with
    # the naive floor check (S4). See PLSPredictBlockVerdict.overall_verdict
    # for the block-level aggregate across this indicator's whole
    # construct, and _predict_verdict (engine.py) for the exact rule —
    # note the naive-benchmark component is an extension of Shmueli et
    # al. (2019), not part of the original paper's rule.
    verdict: Optional[str] = None


class PLSPredictBlockVerdict(BaseModel):
    """
    v1.1 (S5) — Block-level PLSpredict verdict for one endogenous LV's
    indicator block. Aggregates that block's PLSPredictEntry rows via the
    Shmueli et al. (2019) majority-rule decision rule, extended with the
    S4 naive-benchmark floor check (see _predict_verdict in engine.py).
    """
    lv: str
    n_indicators: int
    n_beats_lm: int               # indicators where PLS RMSE < LM RMSE
    n_beats_naive: int            # indicators where PLS RMSE < naive RMSE
    overall_verdict: str          # "high predictive power" | "medium" | "low" | "lacks predictive relevance"
    entries: List[PLSPredictEntry] = []   # this block's rows from `plspredict`


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
    # v1.1 (S5) — one block-level verdict per endogenous LV present in
    # `plspredict`, aggregated from that LV's PLSPredictEntry rows.
    overall_verdict: Optional[List[PLSPredictBlockVerdict]] = None



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

    ``significant`` reflects whichever method was requested as the primary
    ``mga_method`` on the parent :class:`MGAResult` (default ``"bootstrap"``,
    the percentile-CI test — significant when the 95% CI excludes 0). All
    three p-values below are always computed regardless of the primary
    method, so the analyst can compare them side by side (v1.1, S1).
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
    significant: bool            # per mga_method (see MGAResult.mga_method)

    # v0.9 — bootstrap-CI test's own two-tailed empirical p-value
    p_value_bootstrap: Optional[float] = None

    # v1.1 (S1) — Henseler et al. (2009) non-parametric MGA: two-tailed
    # p-value derived from the proportion of bootstrap draws of group A
    # that exceed every draw of group B (and vice versa).
    p_value_henseler: Optional[float] = None

    # Parametric PLS-MGA test (Sarstedt, Henseler & Ringle 2011), corrected
    # for unequal group variances/sample sizes via Welch-Satterthwaite:
    # t = (b_A - b_B) / sqrt(SE_A^2 + SE_B^2), df via Welch-Satterthwaite.
    p_value_parametric: Optional[float] = None
    df_welch: Optional[float] = None   # Welch-Satterthwaite degrees of freedom


class MGAResult(BaseModel):
    """
    Multi-Group Analysis result.

    Includes per-group fit, pairwise bootstrap path-difference CIs,
    and (for 2-group analyses) MICOM measurement-invariance results.
    """
    model_config = ConfigDict(extra="ignore")

    grouping_variable: str
    groups: List[str]
    bootstrap_n: int
    group_results: List[MGAGroupResult]
    path_differences: List[MGAPathDiff]
    micom: Optional[MICOMResult] = None
    # v1.1 (S1) — which of the three methods ("bootstrap" | "henseler" |
    # "parametric") determines MGAPathDiff.significant. All three p-values
    # are always reported on every path regardless of this choice.
    mga_method: str = "bootstrap"
    warnings: List[str] = []

# ── v0.7 schemas ──────────────────────────────────────────────────────────────

class SimpleSlope(BaseModel):
    """
    Conditional effect of the predictor on outcome at one level of the moderator.
    Slope = β_X + β_XM × moderator_value.
    """
    moderator_level: str          # "low (−1 SD)" | "mean (0)" | "high (+1 SD)"
    moderator_value: float        # actual mean-centred value used
    slope: float                  # conditional β
    ci_lower_95: Optional[float] = None
    ci_upper_95: Optional[float] = None
    significant: Optional[bool]  = None   # CI excludes 0


class ModerationTerm(BaseModel):
    """
    One interaction effect: IV × Moderator → Outcome.
    Holds the interaction β, Δ R², f², and simple slopes.
    """
    iv:               str
    moderator:        str
    outcome:          str
    interaction_col:  str          # product column name added to df

    beta_iv:           float
    beta_moderator:    float
    beta_interaction:  float
    ci_lower_95:       Optional[float] = None
    ci_upper_95:       Optional[float] = None
    significant:       bool = False

    r2_with:           float       # R² of outcome with interaction term
    r2_without:        float       # R² without interaction term
    delta_r2:          float       # R²_with − R²_without
    f2_interaction:    float       # Cohen's f² = ΔR² / (1 − R²_with)

    simple_slopes:     List[SimpleSlope] = []


class ModerationResult(BaseModel):
    """Full moderation analysis output."""
    model_config = ConfigDict(extra="ignore")

    algorithm:         str
    n_obs:             int
    bootstrap_n:       int
    moderation_terms:  List[ModerationTerm]
    parameters:        List[PathParameter]  # full model parameters incl. interaction
    fit:               FitIndices
    warnings:          List[str] = []


# ── IPMA ──────────────────────────────────────────────────────────────────────

class IPMAIndicatorEntry(BaseModel):
    lv: str
    indicator: str
    importance: float    # total_effect_on_target * |outer_loading|
    performance: float   # rescaled indicator mean 0-100


class IPMAEntry(BaseModel):
    """
    One construct in the Importance-Performance Map.

    Importance  = total effect of this construct on the target LV
                  (direct + all indirect paths).
    Performance = mean composite score rescaled to 0–100.
    """
    lv:          str
    importance:  float    # total effect on target_lv  (0–1 for standardised)
    performance: float    # rescaled mean composite (0–100)


class IPMAResult(BaseModel):
    """
    Importance-Performance Map Analysis result.
    Entries are sorted by importance descending.
    """
    model_config = ConfigDict(extra="ignore")

    target_lv:   str
    entries:     List[IPMAEntry]
    scale_min:   float            # theoretical or observed scale minimum used
    scale_max:   float            # theoretical or observed scale maximum used
    algorithm:   str
    warnings:    List[str] = []
    indicator_entries: list = []   # list[IPMAIndicatorEntry]

    # v1.1 — Indicator-level IPMA quadrant chart (B2). SVG is the primary
    # format for the web frontend; PNG (base64-encoded) is for DOCX/PDF
    # embedding — see export_docx._build_ipma_section and
    # export_pdf._build_ipma_section. Both are None if chart generation
    # produced no plottable points or failed (see `warnings` for why).
    chart_svg: Optional[str] = None   # inline <svg>...</svg> markup
    chart_png: Optional[str] = None   # base64-encoded PNG bytes



# ── NCA ───────────────────────────────────────────────────────────────────────

class NCAEntry(BaseModel):
    """
    Necessary Condition Analysis result for one IV → DV pair.

    Dul (2016, 2020) CE-FDH and CR-FDH ceiling lines;
    effect size d = ceiling zone / scope.
    """
    iv:            str
    dv:            str
    n_obs:         int

    # CE-FDH (staircase ceiling)
    ce_fdh_d:      float
    ce_fdh_label:  str            # "negligible" | "small" | "medium" | "large"
    ce_fdh_p:      Optional[float] = None   # permutation p-value

    # CR-FDH (linear regression ceiling)
    cr_fdh_d:      float
    cr_fdh_label:  str
    cr_fdh_slope:  float
    cr_fdh_intercept: float
    cr_fdh_p:      Optional[float] = None

    significant:   bool           # max(ce_fdh_p, cr_fdh_p) < 0.05

    # Ceiling line points for frontend scatter plot (sampled to ≤ 200 pts)
    scatter_x:     List[float] = []
    scatter_y:     List[float] = []
    ceiling_x:     List[float] = []
    ceiling_y:     List[float] = []

    bottleneck_table: Optional[list] = None   # list[NCABottleneckRow]


class NCABottleneckRow(BaseModel):
    """Dul (2016) bottleneck table row: minimum X required by the ceiling at a given Y percentile."""
    y_percentile: float     # e.g. 10 = 10th percentile of Y scope
    y_value: float          # actual Y value at that percentile of the scope
    x_required: float       # minimum X that ceiling requires at this Y
    x_percentile: float     # where x_required falls in the observed X distribution


class NCAResult(BaseModel):
    """Full NCA output across all structural IV → DV pairs."""
    model_config = ConfigDict(extra="ignore")

    entries:         List[NCAEntry]
    n_permutations:  int
    warnings:        List[str] = []
    # v1.1 — plain-English ceiling-line annotations (A17). Up to two
    # sentences per NCAEntry — CE-FDH d then CR-FDH d, in that order, each
    # via engine_utils.annotate_fit_index("CE-FDH d", entry.ce_fdh_d) /
    # annotate_fit_index("CR-FDH d", entry.cr_fdh_d). "Ceiling-line ratio"
    # in the ticket refers to these d values (ceiling zone / scope).
    annotations: List[str] = []


# ── NCA-ESSE: Effect Size Sensitivity Extension (v0.9) ────────────────────────
# Becker, J.-M., Richter, N. F., Ringle, C. M., & Sarstedt, M. (2026).
# Must-have, or maybe not? A sensitivity-based extension to necessary condition
# analysis. Journal of Business Research, 206, 115920.
# https://doi.org/10.1016/j.jbusres.2025.115920  (CC BY 4.0)


class NCAESSEThresholdPoint(BaseModel):
    """One ECDF-threshold step in the sensitivity sweep for one IV→DV pair."""
    threshold:         float
    pct_excluded:      float
    empirical_d:       float
    theoretical_d:     float
    delta_empirical:   Optional[float] = None   # Δ vs. previous threshold step
    delta_theoretical: Optional[float] = None
    delta_diff:        Optional[float] = None   # delta_empirical − delta_theoretical
    p_value:           Optional[float] = None   # permutation p-value at this threshold
    p_value_adjusted:  Optional[float] = None   # BH-adjusted p-value
    significant:       bool = False


class NCAESSEEntry(BaseModel):
    """NCA-ESSE sensitivity curve for one structural IV → DV pair."""
    iv:                      str
    dv:                      str
    n_obs:                   int
    thresholds:              List[NCAESSEThresholdPoint]
    recommended_threshold:   Optional[float] = None
    recommended_effect_size: Optional[float] = None
    recommended_label:       Optional[str]   = None
    ceiling_x:               List[float]     = []
    ceiling_y:               List[float]     = []
    warnings:                List[str]       = []


class NCAESSEResult(BaseModel):
    """Top-level container returned by compute_nca_esse()."""
    model_config = ConfigDict(extra="ignore")

    entries:          List[NCAESSEEntry]
    threshold_range:  List[float]
    benchmark:        str = "joint_uniform"
    n_permutations:   int
    n_benchmark_reps: int
    warnings:         List[str] = []


# ── Moderated Mediation (v0.7) ────────────────────────────────────────────────

class ConditionalIndirectEffect(BaseModel):
    """
    Indirect effect of X on Y through M at one specific level of moderator W.

    IE(w) = (a + a₃·w) × b        (a-path moderation)
          = a × (b + b₃·w)        (b-path moderation)
    Significant when the 95 % bootstrap CI excludes zero.
    """
    moderator_level: str           # "low (−1 SD)" | "mean (0)" | "high (+1 SD)"
    moderator_value: float         # actual SD-scaled value used (e.g. −1.23)
    indirect_effect: float         # point-estimate IE at this W level
    ci_lower_95:     Optional[float] = None
    ci_upper_95:     Optional[float] = None
    significant:     bool = False  # True when CI excludes 0


class ModMediationPath(BaseModel):
    """
    One complete X → M → Y chain with moderation on either the a- or b-path.

    Fields
    ------
    x / m / y / w          : variable names
    moderated_path          : "a" (W moderates X→M) | "b" (W moderates M→Y)
    a_path                  : β for X → M
    b_path                  : β for M → Y
    c_prime                 : direct X → Y path (partial effect)
    a3_interaction          : β for the X*W → M product term (a-path only)
    b3_interaction          : β for the M*W → Y product term (b-path only)
    imm                     : Index of Moderated Mediation (a₃·b or a·b₃)
    imm_ci_lower/upper_95   : bootstrap 95 % CI for IMM
    imm_significant         : True when CI excludes 0
    conditional_effects     : IE at W = −1 SD, mean (0), +1 SD
    """
    x: str
    m: str
    y: str
    w: str
    moderated_path:   str          # "a" | "b"

    a_path:           float        # X → M
    b_path:           float        # M → Y
    c_prime:          float        # direct X → Y

    a3_interaction:   Optional[float] = None   # interaction term on a-path
    b3_interaction:   Optional[float] = None   # interaction term on b-path

    imm:              float        # Index of Moderated Mediation (point estimate)
    imm_ci_lower_95:  Optional[float] = None
    imm_ci_upper_95:  Optional[float] = None
    imm_significant:  bool = False

    conditional_effects: List[ConditionalIndirectEffect] = []


class ModMediationResult(BaseModel):
    """
    Full Moderated Mediation / Conditional Process Analysis output.

    Edwards & Lambert (2007); Hayes (2018, Chapters 11–14).
    One ModMediationPath entry per detected X→M→Y chain.
    """
    model_config = ConfigDict(extra="ignore")

    algorithm:   str
    n_obs:       int
    bootstrap_n: int
    paths:       List[ModMediationPath]
    parameters:  List[PathParameter]   # full model parameter table
    fit:         FitIndices
    warnings:    List[str] = []


# ── Nonlinear (Polynomial) Effects (v0.7) ─────────────────────────────────────

class NonlinearEntry(BaseModel):
    path: str
    base_var: str
    outcome: str
    beta_linear: float
    beta_quadratic: float
    r2_linear: float
    r2_augmented: float
    delta_r2: float
    delta_f2: float
    ci_lower_95: Optional[float] = None
    ci_upper_95: Optional[float] = None
    significant: bool = False


class NonlinearResult(BaseModel):
    entries: List[NonlinearEntry]
    algorithm: str
    bootstrap_n: int
    warnings: List[str] = []


# ── Gaussian Copula Endogeneity Correction (v0.8) ─────────────────────────────

class CopulaEntry(BaseModel):
    variable: str
    normality_stat: float          # KS or Shapiro-Wilk statistic
    normality_p: float             # p-value — low = non-normal (copula valid)
    copula_coef: Optional[float] = None
    copula_ci_lower_95: Optional[float] = None
    copula_ci_upper_95: Optional[float] = None
    copula_significant: bool = False
    delta_r2: Optional[float] = None
    f2_copula: Optional[float] = None
    corrected_paths: Dict[str, float] = {}  # outcome: adjusted beta
    original_paths:  Dict[str, float] = {}


class GaussianCopulaResult(BaseModel):
    entries: List[CopulaEntry]
    algorithm: str
    n_obs: int
    bootstrap_n: int
    warnings: List[str] = []


# ── FIMIX-PLS (v0.8) ──────────────────────────────────────────────────────────

class FIMIXSegment(BaseModel):
    segment_id: int
    size: int
    proportion: float                    # pi_k — mixing weight
    path_coefficients: Dict[str, float]  # "lhs~rhs": coef
    r_squared: Dict[str, float]          # outcome: R²


class FIMIXSolution(BaseModel):
    k: int
    log_likelihood: float
    aic: float
    bic: float
    caic: float
    relative_entropy: float              # R_E — separation quality 0–1
    segments: List[FIMIXSegment]


class FIMIXResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    solutions: List[FIMIXSolution]
    recommended_k: int                   # k minimising CAIC
    algorithm: str = "fimix-pls"
    n_obs: int
    warnings: List[str] = []


# ── PLS-POS (v0.8) ────────────────────────────────────────────────────────────

class PLSPOSSegment(BaseModel):
    segment_id: int
    size: int
    path_coefficients: Dict[str, float]  # "lhs~rhs": coef
    r_squared: Dict[str, float]          # outcome: R²
    stability: float                     # proportion of bootstrap runs with same assignment (0–1)


class PLSPOSResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    k: int
    segments: List[PLSPOSSegment]
    fimix_comparison: Optional[Dict[str, Any]] = None  # FIMIX vs POS coef table
    algorithm: str = "pls-pos"
    n_obs: int
    warnings: List[str] = []


# ── Robustness Checks wrapper (v0.8) ──────────────────────────────────────────

class RobustnessChecks(BaseModel):
    nonlinear:      Optional["NonlinearResult"]       = None
    fimix:          Optional["FIMIXResult"]           = None
    plspos:         Optional["PLSPOSResult"]          = None
    copula:         Optional["GaussianCopulaResult"]  = None
    copula_warning: Optional[str]                     = None


# ── fsQCA schemas (v1.0) ──────────────────────────────────────────────────────
# Ragin, C. C. (2008). Redesigning Social Inquiry. University of Chicago Press.
# Schneider, C. Q., & Wagemann, C. (2012). Set-Theoretic Methods for the
#   Social Sciences. Cambridge University Press.


class NecessityEntry(BaseModel):
    """Necessity analysis result for one condition."""
    condition:   str
    coverage:    float
    consistency: float
    label:       str      # "Necessary" | "Near-Necessary" | "Not Necessary"


class TruthTableRow(BaseModel):
    """One row of the fsQCA truth table (one of 2^k configurations).

    Condition columns (e.g. ``X1``, ``X2``) are stored as extra fields so that
    the serialised JSON exposes each condition's crisp value (0 or 1) as a direct
    top-level key — required by the JS ``_renderFsQCAResults`` function.
    """
    model_config = ConfigDict(extra="allow")

    configuration: str    # binary string, e.g. "101"
    n:             int    # cases primarily assigned to this configuration
    consistency:   float  # PRI score for cases assigned here
    outcome:       int    # 1 if passes freq & consist thresholds; else 0


class FsQCAConfigTerm(BaseModel):
    """One prime implicant term in a minimized fsQCA solution."""
    configuration:    str    # QCA notation, e.g. "X1*~X2*X3"
    raw_coverage:     float
    unique_coverage:  float
    consistency:      float


class FsQCASolution(BaseModel):
    """Minimized Boolean solution (complex, parsimonious, or intermediate)."""
    solution_type:        str               # "complex" | "parsimonious" | "intermediate"
    terms:                List[FsQCAConfigTerm]
    solution_coverage:    float
    solution_consistency: float


class BubbleChartPoint(BaseModel):
    """One XY data point for the fuzzy-set coincidence (bubble) chart."""
    case_id:      int    # zero-based row index in the fuzzy DataFrame
    condition:    str    # condition column name
    x_membership: float  # calibrated set membership in condition  (0–1)
    y_membership: float  # calibrated set membership in outcome    (0–1)


class FsQCAResult(BaseModel):
    """Full fsQCA output returned by run_fsqca().

    Top-level ``minimized_solution`` / ``consistency`` / ``coverage`` fields
    mirror the complex-solution values and are consumed directly by the
    ``_renderFsQCAResults`` JavaScript function.  The full three-solution
    breakdown is still available under ``solutions``.
    """
    model_config = ConfigDict(extra="ignore")

    outcome:           str
    conditions:        List[str]
    n_obs:             int
    necessity:         List[NecessityEntry]
    truth_table:       List[TruthTableRow]
    solutions:         List[FsQCASolution]
    # ── UI-consumed shortcut fields (complex solution) ─────────────────────
    minimized_solution: Optional[str]   = None   # prime-implicant string, " + "-joined
    consistency:        Optional[float] = None   # solution-level PRI consistency
    coverage:           Optional[float] = None   # solution-level coverage
    # ──────────────────────────────────────────────────────────────────────
    warnings:          List[str]              = []
    bubble_chart_data: List[BubbleChartPoint] = []


# ── Bayesian SEM schemas (v1.1, A10) ─────────────────────────────────────────
# Lee, S.-Y. (2007). Structural Equation Modeling: A Bayesian Approach. Wiley.
# Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd ed. — split R-hat.
# Vehtari, A., et al. (2021). Rank-normalization, folding, and localization:
#   An improved R-hat for MCMC. Bayesian Analysis. — bulk-ESS.
# Chen, M.-H., & Shao, Q.-M. (1999). Monte Carlo estimation of Bayesian
#   credible and HPD intervals. J. Comp. Graph. Statistics, 8(1), 69-92.


class BayesianParameterEntry(BaseModel):
    """One free parameter's posterior summary from fit_bayesian_sem().

    ``name`` is the canonical lavaan-style key used to key the ``priors``
    dict in A9 and to match ModelResult.parameters' (lhs, op, rhs) identity
    elsewhere in the codebase, e.g. "Trust=~t1" (loading), "Sat~Trust"
    (structural path), "Trust~~Trust" (factor/disturbance variance),
    "t1~~t1" (indicator residual variance).
    """
    name: str
    op:   str     # "=~" | "~" | "~~"
    lhs:  str
    rhs:  str
    posterior_mean:   float
    posterior_median: float
    posterior_sd:     float
    # 95% HIGHEST POSTERIOR DENSITY interval — the shortest interval containing
    # 95% of posterior mass (Chen & Shao 1999), NOT a symmetric mean ± 1.96*sd
    # interval and NOT the percentile interval used for BootstrapParameter's
    # ci_lower_95/ci_upper_95 elsewhere in this file. For a skewed posterior
    # (routine for variance parameters) the two intervals diverge meaningfully;
    # HPD is the one that actually has 95% posterior mass inside it.
    hpd_lower: float
    hpd_upper: float
    r_hat:    float   # split R-hat (Gelman-Rubin); should be <= 1.01 at convergence
    ess_bulk: float    # rank-normalized bulk effective sample size (Vehtari et al. 2021)


class BayesianResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parameters: List[BayesianParameterEntry]
    n_chains:  int
    n_samples: int              # post-warmup draws retained per chain
    converged: bool              # True iff max(r_hat) <= 1.01 across all parameters
    convergence_warnings: List[str] = []   # parameter names that failed the R-hat check


class ParameterPosteriorDensity(BaseModel):
    """Binned posterior density for one parameter (A11): 50 bins, sized for
    the existing frontend bubble-chart/histogram component. x = bin centers,
    y = density heights. Never raw sample arrays."""
    name: str
    x: List[float]
    y: List[float]


class BayesianSemResponse(BaseModel):
    """Top-level response body for POST /bayesian-sem."""
    model_config = ConfigDict(extra="ignore")

    result: BayesianResult
    posterior_density: List[ParameterPosteriorDensity]


# ── General Latent Class / Finite Mixture engine (v1.1, A12–A15) ──────────────
# Reuses the FIMIX-PLS EM scaffolding (fimix.py) but operates on raw indicator
# columns directly. See app/engine_lca.py.

class LCAFitRow(BaseModel):
    """One row of the K-selection fit table (mirrors FIMIXSolution's fit
    fields so recommend_k() in fimix.py can be reused unmodified)."""
    k:                 int
    log_likelihood:    float
    aic:               float
    bic:               float
    caic:              float
    relative_entropy:  float          # R_E — separation quality 0–1


class LCAClassParameters(BaseModel):
    """Class-specific (or constraint-group-pooled) parameter estimates.

    ``parameters`` keys are parameter names in the naming convention used by
    the relevant mode's M-step:
      - segmentation:        "mean_<indicator>", "var_<indicator>"
      - mixture_regression:  "<iv_col>" (coefficient), "sigma2"
      - mixture_factor:      "mean_<indicator>", "loading_<indicator>",
                              "uniq_<indicator>"
    """
    class_id:   int
    size:       int
    proportion: float
    parameters: Dict[str, float]


class LCAResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    algorithm:            str = "lca"
    mode:                  str                      # "segmentation" | "mixture_regression" | "mixture_factor"
    n_obs:                 int
    indicator_cols:        List[str]
    class_sizes:           Dict[int, Dict[int, int]]   # {k: {class_id: size}}
    fit_table:             List[LCAFitRow]
    recommended_k:         int
    per_case_membership:   Dict[int, List[List[float]]]  # {k: [[p_class0, p_class1, ...], ...]} row-ordered
    parameters:            Dict[int, List[LCAClassParameters]]  # {k: [per-class params]}
    equality_constraints:  List[str] = []
    known_class_col:       Optional[str] = None
    warnings:              List[str] = []


# ── v1.1 (S2) — Confirmatory Tetrad Analysis (CTA-PLS) ─────────────────────────

class CTATetradEntry(BaseModel):
    """
    One bootstrapped tetrad test for a single quartet of indicators within
    a reflective LV block (Bollen & Ting 2000).

    A tetrad is the algebraic difference of two covariance products,
    e.g. τ = σ(w,x)·σ(y,z) − σ(w,z)·σ(x,y). Under a single-factor
    (reflective/congeneric) measurement model every tetrad should vanish
    (≈ 0) in the population. A tetrad whose bootstrap CI excludes 0 is
    evidence *against* the reflective specification for that block.
    """
    lv_name: str
    indicators: List[str]        # the four indicators (w, x, y, z) in this tetrad
    pairing: str                 # which of the 3 covariance-product pairings, e.g. "wx.yz"
    value: float                 # observed tetrad value (full sample)
    ci_lower_95: float
    ci_upper_95: float
    vanishes: bool                # True (supports reflective) when 0 ∈ [ci_lower_95, ci_upper_95]


class CTALVResult(BaseModel):
    """CTA-PLS verdict for one reflective LV block."""
    lv_name: str
    n_indicators: int
    n_tetrads_tested: int         # size of the non-redundant tetrad set (Bollen & Ting 2000)
    n_significant: int            # tetrads whose CI excludes 0 (non-vanishing)
    verdict: str                  # "supports reflective" | "consider formative respecification"
    tetrads: List[CTATetradEntry] = []


class CTAResult(BaseModel):
    """Full Confirmatory Tetrad Analysis output across all eligible reflective LVs."""
    model_config = ConfigDict(extra="ignore")

    bootstrap_n: int
    lv_results: List[CTALVResult]
    warnings: List[str] = []


# ── v1.1 (A16) — Multi-group CB-SEM with equality constraints ─────────────────

class MultigroupCBSEMFit(BaseModel):
    """Fit summary for one side (free or constrained) of the multi-group test."""
    chi_square: Optional[float] = None
    df: Optional[int] = None
    n_free_parameters: Optional[int] = None
    per_group: Dict[str, FitIndices] = {}   # per-group CFI/RMSEA/SRMR/etc (free fit only)
    parameters: List[PathParameter] = []    # pooled/shared + per-group estimates, where available


class MultigroupCBSEMResult(BaseModel):
    """
    Multi-group CB-SEM likelihood-ratio test result (A16).

    ``free``        : all parameters estimated separately per group (configural).
    ``constrained``  : parameters named in ``equality_constraints`` forced equal
                        across groups; every other parameter remains free per group.
    A significant LR chi-square difference test (``lr_p_value`` < .05) means
    the constrained (equal-parameter) model fits significantly worse than the
    free model — i.e. the constrained equality is rejected.
    """
    model_config = ConfigDict(extra="ignore")

    group_col: str
    groups: List[str]
    equality_constraints: List[str]
    free: MultigroupCBSEMFit
    constrained: MultigroupCBSEMFit
    lr_chi_square: Optional[float] = None
    lr_df: Optional[int] = None
    lr_p_value: Optional[float] = None
    constrained_rejected: Optional[bool] = None   # True when lr_p_value < .05
    conclusion: str = ""
    warnings: List[str] = []


# Resolve all forward references now that every class is defined.
ModelResult.model_rebuild()
