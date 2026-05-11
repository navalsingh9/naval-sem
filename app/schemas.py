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

    # Measurement validity metrics
    ave: Optional[Dict[str, float]] = None                          # AVE per LV
    composite_reliability: Optional[Dict[str, float]] = None        # ρc per LV
    cronbach_alpha: Optional[Dict[str, float]] = None               # α per LV
    fornell_larcker: Optional[Dict[str, Dict[str, float]]] = None   # √AVE on diag, r off-diag
    fornell_larcker_pass: Optional[bool] = None                     # True when all √AVE > off-diag r

    # Fit verdict helpers
    cfi_acceptable: Optional[bool] = None    # CFI >= 0.90
    cfi_good: Optional[bool] = None          # CFI >= 0.95
    rmsea_acceptable: Optional[bool] = None  # RMSEA <= 0.08
    rmsea_good: Optional[bool] = None        # RMSEA <= 0.06
    srmr_good: Optional[bool] = None         # SRMR <= 0.08


class BootstrapResult(BaseModel):
    n_samples: int
    parameters: List[Dict[str, Any]]    # Same shape as PathParameter + bs_se, bs_ci_lower/upper
    converged_pct: float                # % of bootstrap samples that converged


class HTMTEntry(BaseModel):
    construct_a: str
    construct_b: str
    htmt: float
    acceptable: bool    # HTMT < 0.90


class HTMTResult(BaseModel):
    matrix: List[HTMTEntry]
    all_acceptable: bool


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
    warnings: List[str] = []
