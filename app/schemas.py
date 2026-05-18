"""
Pydantic schemas for all API responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PathParameter(BaseModel):
    lhs: str
    op: str            # =~ (measurement) or ~ (structural)
    rhs: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    significant: bool  # p < 0.05


class FitIndices(BaseModel):
    cfi: Optional[float] = None
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
    indicator: str
    estimate: float          # point estimate from full-data fit
    bs_mean: float           # mean across bootstrap samples
    bs_se: float             # bootstrap standard error
    ci_lower_95: float
    ci_upper_95: float
    t_stat: Optional[float] = None   # estimate / bs_se
    significant: bool                # CI excludes zero


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

