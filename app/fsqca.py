"""
app/fsqca.py — fuzzy-set Qualitative Comparative Analysis (fsQCA) engine.
NAVAL-SEM v1.0

Public API
----------
calibrate_fuzzy         : Raw → fuzzy membership (direct or indirect method).
compute_necessity       : Necessity analysis for each condition.
build_truth_table       : Crisp truth-table construction with PRI consistency.
minimize_boolean        : Quine-McCluskey minimization → complex / parsimonious /
                          intermediate solutions.
generate_coincidence_data: XY bubble-chart data for coincidence plots.
run_fsqca               : Top-level orchestrator; returns FsQCAResult.

References
----------
Ragin, C. C. (2008). *Redesigning Social Inquiry*. University of Chicago Press.
Schneider, C. Q., & Wagemann, C. (2012). *Set-Theoretic Methods for the Social
    Sciences*. Cambridge University Press.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── internal constants ────────────────────────────────────────────────────────
_LOG19 = float(np.log(19.0))          # logit(0.95) ≈ 2.944 — used as "full" anchor


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _interp_logit(x: float, anchors: list[tuple[float, float]]) -> float:
    """
    Piecewise-linear interpolation in log-odds space.

    Parameters
    ----------
    x       : Raw value.
    anchors : Sorted list of (raw_value, logit) pairs (at least two).

    Returns
    -------
    float log-odds (unclamped).
    """
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, l0), (x1, l1) in zip(anchors, anchors[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return l0 + t * (l1 - l0)
    return anchors[-1][1]


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 1 — calibrate_fuzzy
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_fuzzy(
    series: pd.Series,
    method: str = "indirect",
    crossover: Optional[float] = None,
    threshold_in: Optional[float] = None,
    threshold_out: Optional[float] = None,
) -> pd.Series:
    """
    Calibrate a raw numeric series to fuzzy set membership scores.

    Parameters
    ----------
    series        : Raw numeric pd.Series.
    method        : ``"direct"`` — log-odds transformation (Ragin 2008);
                    ``"indirect"`` — percentile-based.
    crossover     : (direct) Raw value → 0.5 membership.
    threshold_in  : (direct) Raw value → full membership (≈ 0.95).
    threshold_out : (direct) Raw value → full non-membership (≈ 0.05).

    Returns
    -------
    pd.Series clipped to [0.001, 0.999].
    Notes
    -----
    0 and 1 are reserved for logical operations and must not appear in output.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)

    if method == "direct":
        if crossover is None or threshold_in is None or threshold_out is None:
            raise ValueError(
                "calibrate_fuzzy (direct): crossover, threshold_in, and "
                "threshold_out are all required."
            )
        # Three log-odds anchors — sort by raw value so interpolation is monotone.
        # Sign of LOG19 follows whether threshold_in represents "more" or "less".
        if threshold_in >= crossover:
            anchors = sorted(
                [
                    (float(threshold_out), -_LOG19),
                    (float(crossover), 0.0),
                    (float(threshold_in), _LOG19),
                ],
                key=lambda t: t[0],
            )
        else:
            # Inverted scale (higher raw value = lower membership)
            anchors = sorted(
                [
                    (float(threshold_out), _LOG19),
                    (float(crossover), 0.0),
                    (float(threshold_in), -_LOG19),
                ],
                key=lambda t: t[0],
            )

    elif method == "indirect":
        p5  = float(np.nanpercentile(s, 5))
        p50 = float(np.nanpercentile(s, 50))
        p95 = float(np.nanpercentile(s, 95))
        anchors = sorted(
            [(p5, -_LOG19), (p50, 0.0), (p95, _LOG19)],
            key=lambda t: t[0],
        )

    else:
        raise ValueError(
            f"calibrate_fuzzy: unknown method '{method}'. Use 'direct' or 'indirect'."
        )

    result = s.map(
        lambda x: _sigmoid(_interp_logit(x, anchors)) if pd.notna(x) else np.nan
    )
    return result.clip(lower=0.001, upper=0.999)


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 2 — compute_necessity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_necessity(
    fuzzy_df: pd.DataFrame,
    outcome_col: str,
    condition_cols: list[str],
) -> list:
    """
    Necessity analysis: for each condition X and outcome Y,

        coverage    = Σ min(X, Y) / Σ Y
        consistency = Σ min(X, Y) / Σ X

    Label rules:
        "Necessary"      — consistency ≥ 0.90 AND coverage ≥ 0.50
        "Near-Necessary" — consistency ≥ 0.80
        "Not Necessary"  — otherwise

    Returns
    -------
    list[NecessityEntry]
    """
    from app.schemas import NecessityEntry

    Y = fuzzy_df[outcome_col].values.astype(float)
    entries: list = []

    for col in condition_cols:
        X = fuzzy_df[col].values.astype(float)
        mins = np.minimum(X, Y)

        sum_Y = float(np.sum(Y))
        sum_X = float(np.sum(X))

        coverage    = float(np.sum(mins) / sum_Y) if sum_Y > 0 else 0.0
        consistency = float(np.sum(mins) / sum_X) if sum_X > 0 else 0.0

        if consistency >= 0.90 and coverage >= 0.50:
            label = "Necessary"
        elif consistency >= 0.80:
            label = "Near-Necessary"
        else:
            label = "Not Necessary"

        entries.append(
            NecessityEntry(
                condition=col,
                coverage=round(coverage, 4),
                consistency=round(consistency, 4),
                label=label,
            )
        )

    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Truth-table internals
# ═══════════════════════════════════════════════════════════════════════════════

def _config_membership(
    fuzzy_df: pd.DataFrame,
    condition_cols: list[str],
    config_bits: tuple[int, ...],
) -> np.ndarray:
    """
    Fuzzy set membership of every case in a configuration.

    For bit=1 : use raw condition membership.
    For bit=0 : use 1 − condition membership (negation).
    Intersection = element-wise minimum.
    """
    n = len(fuzzy_df)
    mem = np.ones(n, dtype=float)
    for bit, col in zip(config_bits, condition_cols):
        x = fuzzy_df[col].values.astype(float)
        mem = np.minimum(mem, x if bit == 1 else 1.0 - x)
    return mem


def _pri_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Proportional Reduction in Inconsistency (PRI) consistency score.

        PRI = ( Σ min(X,Y) − Σ min(X, 1−Y, Y) )
              / ( Σ X − Σ min(X, 1−Y, Y) )

    Returns 0.0 when denominator ≤ 0.
    """
    shared    = np.minimum(x, y)
    ambiguous = np.minimum(x, np.minimum(1.0 - y, y))
    num = float(np.sum(shared)    - np.sum(ambiguous))
    den = float(np.sum(x)         - np.sum(ambiguous))
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 3 — build_truth_table
# ═══════════════════════════════════════════════════════════════════════════════

def build_truth_table(
    fuzzy_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    freq_threshold: int = 1,
    consist_threshold: float = 0.75,
) -> list:
    """
    Build the truth table using crisp assignment.

    Each case is assigned to the configuration in which it holds the
    highest fuzzy membership (argmax over all 2^k rows).  PRI consistency
    is then computed from the fuzzy membership scores of the assigned cases.

    Returns
    -------
    list[TruthTableRow]  — all 2^k rows, sorted lexicographically.
    """
    from app.schemas import TruthTableRow

    k = len(condition_cols)
    Y = fuzzy_df[outcome_col].values.astype(float)
    all_configs: list[tuple[int, ...]] = list(itertools.product([0, 1], repeat=k))

    # Build (n_cases × 2^k) membership matrix
    mem_matrix = np.stack(
        [_config_membership(fuzzy_df, condition_cols, cfg) for cfg in all_configs],
        axis=1,
    )                                                   # shape: (n_cases, 2^k)

    # Crisp assignment: each case → highest-membership configuration
    assigned = np.argmax(mem_matrix, axis=1)            # shape: (n_cases,)

    rows: list = []
    for idx, cfg in enumerate(all_configs):
        mask       = assigned == idx
        n_assigned = int(np.sum(mask))
        cfg_str    = "".join(str(b) for b in cfg)

        if n_assigned == 0:
            consist = 0.0
        else:
            consist = _pri_score(mem_matrix[mask, idx], Y[mask])

        outcome_val = 1 if (consist >= consist_threshold and n_assigned >= freq_threshold) else 0

        rows.append(
            TruthTableRow(
                configuration=cfg_str,
                n=n_assigned,
                consistency=round(consist, 4),
                outcome=outcome_val,
                **{col: int(bit) for col, bit in zip(condition_cols, cfg)},
            )
        )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Quine-McCluskey engine
# ═══════════════════════════════════════════════════════════════════════════════

def _can_combine(a: str, b: str) -> int | None:
    """
    Return the position at which ``a`` and ``b`` differ if they differ in
    exactly one non-dash position; otherwise ``None``.
    """
    diff_pos: int | None = None
    count = 0
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca == "-" or cb == "-":
            if ca != cb:
                return None          # mismatched dashes — cannot combine
        elif ca != cb:
            count += 1
            diff_pos = i
            if count > 1:
                return None
    return diff_pos if count == 1 else None


def _merge(a: str, b: str) -> str:
    """Merge two compatible implicants, replacing the differing bit with '-'."""
    result = list(a)
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            result[i] = "-"
    return "".join(result)


def _quine_mccluskey(minterms: list[str], dontcares: list[str]) -> list[str]:
    """
    Classic Quine-McCluskey prime implicant extraction.

    Parameters
    ----------
    minterms  : Truth-table rows coded outcome=1 (binary strings).
    dontcares : Remainder rows treated as don't-cares (binary strings).

    Returns
    -------
    list of prime implicants (binary strings, possibly with '-' wildcards).
    """
    if not minterms:
        return []

    # Seed the iteration with all minterms + don't-cares
    current: dict[str, bool] = {t: False for t in (set(minterms) | set(dontcares))}
    prime_implicants: list[str] = []

    while True:
        # Group by 1-count (dashes count as 0 for grouping)
        groups: dict[int, list[str]] = {}
        for term in current:
            groups.setdefault(term.count("1"), []).append(term)

        next_round: dict[str, bool] = {}
        covered: set[str] = set()

        for g in sorted(groups):
            if g + 1 not in groups:
                continue
            for t1 in groups[g]:
                for t2 in groups[g + 1]:
                    if _can_combine(t1, t2) is not None:
                        merged = _merge(t1, t2)
                        next_round[merged] = False
                        covered.add(t1)
                        covered.add(t2)

        # Uncombined terms in this round are prime implicants
        for term in current:
            if term not in covered:
                prime_implicants.append(term)

        if not next_round:
            break
        current = next_round

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_pis: list[str] = []
    for pi in prime_implicants:
        if pi not in seen:
            seen.add(pi)
            unique_pis.append(pi)
    return unique_pis


def _covers(prime: str, minterm: str) -> bool:
    """True when ``prime`` covers ``minterm`` (dash = wildcard)."""
    return all(p == "-" or p == m for p, m in zip(prime, minterm))


def _select_cover(prime_implicants: list[str], minterms: list[str]) -> list[str]:
    """
    Select a minimal cover.

    1. Identify essential prime implicants (each covers at least one minterm
       that no other PI covers).
    2. Greedily extend with whichever remaining PI covers the most uncovered
       minterms.
    """
    if not minterms or not prime_implicants:
        return []

    # Coverage map
    mi_to_pis: dict[str, list[str]] = {m: [] for m in minterms}
    for pi in prime_implicants:
        for m in minterms:
            if _covers(pi, m):
                mi_to_pis[m].append(pi)

    selected: list[str] = []
    covered: set[str] = set()

    # Essential PIs
    for m, pis in mi_to_pis.items():
        if len(pis) == 1:
            pi = pis[0]
            if pi not in selected:
                selected.append(pi)

    for pi in selected:
        for m in minterms:
            if _covers(pi, m):
                covered.add(m)

    # Greedy remainder
    remaining_m  = [m for m in minterms  if m not in covered]
    remaining_pi = [pi for pi in prime_implicants if pi not in selected]

    while remaining_m and remaining_pi:
        best = max(remaining_pi, key=lambda pi: sum(_covers(pi, m) for m in remaining_m))
        if not any(_covers(best, m) for m in remaining_m):
            break
        selected.append(best)
        remaining_pi.remove(best)
        remaining_m = [m for m in remaining_m if not _covers(best, m)]

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Fuzzy coverage / consistency for a solution
# ═══════════════════════════════════════════════════════════════════════════════

def _term_fuzzy_membership(
    fuzzy_df: pd.DataFrame,
    condition_cols: list[str],
    prime: str,
) -> np.ndarray:
    """
    Fuzzy membership of each case in a prime implicant.
    Dash positions are don't-cares and do not restrict membership.
    """
    n = len(fuzzy_df)
    mem = np.ones(n, dtype=float)
    for bit, col in zip(prime, condition_cols):
        if bit == "-":
            continue
        x = fuzzy_df[col].values.astype(float)
        mem = np.minimum(mem, x if bit == "1" else 1.0 - x)
    return mem


def _solution_metrics(
    fuzzy_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    selected_primes: list[str],
) -> tuple[float, float, list[dict]]:
    """
    Compute per-term and overall coverage / consistency.

    Returns
    -------
    (solution_coverage, solution_consistency, per_term_list)
    """
    Y     = fuzzy_df[outcome_col].values.astype(float)
    sum_Y = float(np.sum(Y))

    term_mems = [
        _term_fuzzy_membership(fuzzy_df, condition_cols, p) for p in selected_primes
    ]

    if not term_mems:
        return 0.0, 0.0, []

    # Solution = union of all terms
    sol_mem = np.zeros(len(fuzzy_df), dtype=float)
    for tm in term_mems:
        sol_mem = np.maximum(sol_mem, tm)

    sum_min_sol = float(np.sum(np.minimum(sol_mem, Y)))
    sum_sol     = float(np.sum(sol_mem))

    solution_coverage    = round(sum_min_sol / sum_Y  if sum_Y  > 0 else 0.0, 4)
    solution_consistency = round(sum_min_sol / sum_sol if sum_sol > 0 else 0.0, 4)

    per_term: list[dict] = []
    for i, (prime, tm) in enumerate(zip(selected_primes, term_mems)):
        raw_cov = float(np.sum(np.minimum(tm, Y))) / sum_Y if sum_Y > 0 else 0.0

        # Unique coverage = what this term adds beyond all others
        others = np.zeros(len(fuzzy_df), dtype=float)
        for j, om in enumerate(term_mems):
            if j != i:
                others = np.maximum(others, om)
        unique_cov = max(
            0.0,
            (float(np.sum(np.minimum(tm, Y))) - float(np.sum(np.minimum(others, Y)))) / sum_Y
            if sum_Y > 0 else 0.0,
        )

        sum_tm  = float(np.sum(tm))
        consist = float(np.sum(np.minimum(tm, Y))) / sum_tm if sum_tm > 0 else 0.0

        per_term.append(
            {
                "prime":           prime,
                "raw_coverage":    round(raw_cov,    4),
                "unique_coverage": round(unique_cov, 4),
                "consistency":     round(consist,    4),
            }
        )

    return solution_coverage, solution_consistency, per_term


def _prime_to_notation(prime: str, condition_cols: list[str]) -> str:
    """
    Convert a prime implicant string to QCA notation.

    Examples
    --------
    "101" + [X1, X2, X3]  →  "X1*~X2*X3"
    "-01" + [X1, X2, X3]  →  "~X2*X3"
    "---"                  →  "(1)"    (tautology)
    """
    parts = []
    for bit, col in zip(prime, condition_cols):
        if bit == "1":
            parts.append(col)
        elif bit == "0":
            parts.append(f"~{col}")
        # "-" → skip (don't-care)
    return "*".join(parts) if parts else "(1)"


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 4 — minimize_boolean
# ═══════════════════════════════════════════════════════════════════════════════

def minimize_boolean(
    truth_table: list,
    solution_type: str,
    fuzzy_df: pd.DataFrame,
    condition_cols: list[str],
    outcome_col: str,
    directional_expectations: Optional[dict] = None,
) -> object:
    """
    Boolean minimization via the Quine-McCluskey algorithm.

    Parameters
    ----------
    truth_table   : list[TruthTableRow] from :func:`build_truth_table`.
    solution_type : ``"complex"``       — confirmed rows only, no remainders.
                    ``"parsimonious"``  — remainders treated as don't-cares.
                    ``"intermediate"``  — directional expectations:
                                         remainders with ≥ k/2 present conditions
                                         are added as don't-cares.
    fuzzy_df      : Calibrated fuzzy DataFrame.
    condition_cols: Condition column names (same order as truth table bits).
    outcome_col   : Outcome column name.

    Returns
    -------
    FsQCASolution
    """
    from app.schemas import FsQCASolution, FsQCAConfigTerm

    k         = len(condition_cols)
    minterms  = [row.configuration for row in truth_table if row.outcome == 1]
    remainders = [row.configuration for row in truth_table if row.n == 0]

    if solution_type == "complex":
        dontcares: list[str] = []
    elif solution_type == "parsimonious":
        dontcares = remainders
    elif solution_type == "intermediate":
        if directional_expectations:
            # Include remainder if every condition bit matches its expected direction
            def _matches_expectations(config: str) -> bool:
                for ci, col in enumerate(condition_cols):
                    direction = directional_expectations.get(col, "presence")
                    expected_bit = "1" if direction == "presence" else "0"
                    if config[ci] != expected_bit:
                        return False
                return True
            dontcares = [r for r in remainders if _matches_expectations(r)]
        else:
            # Legacy fallback with warning
            import warnings as _w
            _w.warn(
                "minimize_boolean: intermediate solution using heuristic fallback "
                "(>=k/2 present conditions). Pass directional_expectations={'cond_name': "
                "'presence'|'absence'} for the textbook Ragin (2008) intermediate solution.",
                UserWarning, stacklevel=2,
            )
            dontcares = [r for r in remainders if r.count("1") >= k / 2]
    else:
        raise ValueError(f"minimize_boolean: invalid solution_type '{solution_type}'.")

    if not minterms:
        return FsQCASolution(
            solution_type=solution_type,
            terms=[],
            solution_coverage=0.0,
            solution_consistency=0.0,
        )

    pis      = _quine_mccluskey(minterms, dontcares)
    selected = _select_cover(pis, minterms)

    # Fallback: if selection produced nothing (edge case), use all PIs
    if not selected and pis:
        selected = pis[:1]

    sol_cov, sol_consist, per_term = _solution_metrics(
        fuzzy_df, condition_cols, outcome_col, selected
    )

    terms = [
        FsQCAConfigTerm(
            configuration=_prime_to_notation(pt["prime"], condition_cols),
            raw_coverage=pt["raw_coverage"],
            unique_coverage=pt["unique_coverage"],
            consistency=pt["consistency"],
        )
        for pt in per_term
    ]

    return FsQCASolution(
        solution_type=solution_type,
        terms=terms,
        solution_coverage=sol_cov,
        solution_consistency=sol_consist,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 5 — run_fsqca  (top-level orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fsqca(
    df: pd.DataFrame,
    outcome_col: str,
    condition_cols: list[str],
    calibration_params: Optional[dict] = None,
    freq_threshold: int = 1,
    consist_threshold: float = 0.75,
    log_fn: Optional[Callable] = None,
    directional_expectations: Optional[dict] = None,
) -> object:
    """
    Top-level fsQCA orchestrator.

    Parameters
    ----------
    df                : Raw or pre-calibrated DataFrame.
    outcome_col       : Column name of the outcome set.
    condition_cols    : Ordered list of condition column names.
    calibration_params: ``{col: {method, crossover, threshold_in, threshold_out}}``
                        for columns that need calibration; omit columns that are
                        already in [0, 1].  Pass ``{}`` or ``None`` to rely on
                        auto-detection.
    freq_threshold    : Minimum case count for a truth-table row to be non-remainder.
    consist_threshold : Minimum PRI score to code a row as outcome = 1.
    log_fn            : Optional ``(level: str, msg: str) → None`` callback.

    Returns
    -------
    FsQCAResult  — necessity table, truth table, three minimized solutions,
                   and bubble-chart data.
    """
    from app.schemas import FsQCAResult

    def _log(level: str, msg: str) -> None:
        if log_fn:
            log_fn(level, msg)
        logger.info("[fsQCA] %s — %s", level.upper(), msg)

    warnings_out: list[str] = []
    calibration_params = calibration_params or {}

    # ── Validate columns ─────────────────────────────────────────────────────
    all_cols  = [outcome_col] + list(condition_cols)
    missing   = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"fsQCA: columns not found in data: {missing}")

    _log("step", f"Starting: {len(condition_cols)} condition(s), {len(df)} row(s)")

    # ── Calibrate / clip ─────────────────────────────────────────────────────
    fuzzy_df = df[all_cols].copy()
    # Ensure numeric
    for col in all_cols:
        fuzzy_df[col] = pd.to_numeric(fuzzy_df[col], errors="coerce")

    for col in all_cols:
        if col in calibration_params:
            p = calibration_params[col]
            _log("info", f"Calibrating '{col}' via method={p.get('method', 'indirect')}")
            fuzzy_df[col] = calibrate_fuzzy(
                fuzzy_df[col],
                method=p.get("method", "indirect"),
                crossover=p.get("crossover"),
                threshold_in=p.get("threshold_in"),
                threshold_out=p.get("threshold_out"),
            )
        else:
            col_min = fuzzy_df[col].min(skipna=True)
            col_max = fuzzy_df[col].max(skipna=True)
            if pd.isna(col_min) or pd.isna(col_max):
                pass  # handled by dropna below
            elif col_min < 0.0 or col_max > 1.0:
                _log(
                    "warn",
                    f"'{col}' has values outside [0,1] ({col_min:.3f}–{col_max:.3f}); "
                    "applying indirect calibration.",
                )
                warnings_out.append(
                    f"{col}: values outside [0,1] — auto-calibrated (indirect)."
                )
                fuzzy_df[col] = calibrate_fuzzy(fuzzy_df[col], method="indirect")
            else:
                # Already in range — just enforce the (0.001, 0.999) bound
                fuzzy_df[col] = fuzzy_df[col].clip(lower=0.001, upper=0.999)

    # ── Drop missing ─────────────────────────────────────────────────────────
    n_before = len(fuzzy_df)
    fuzzy_df = fuzzy_df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(fuzzy_df)
    if n_dropped > 0:
        msg = f"Dropped {n_dropped} row(s) with missing values after calibration."
        _log("warn", msg)
        warnings_out.append(msg)

    n_obs = len(fuzzy_df)
    if n_obs < 2:
        raise ValueError("fsQCA requires at least 2 complete observations after calibration.")

    # ── Step 1: Necessity ─────────────────────────────────────────────────────
    _log("step", "Necessity analysis")
    necessity = compute_necessity(fuzzy_df, outcome_col, condition_cols)

    # ── Step 2: Truth table ───────────────────────────────────────────────────
    _log("step", "Building truth table")
    tt_rows = build_truth_table(
        fuzzy_df,
        list(condition_cols),
        outcome_col,
        freq_threshold=freq_threshold,
        consist_threshold=consist_threshold,
    )
    n_outcome = sum(r.outcome for r in tt_rows)
    _log("info", f"Truth table: {len(tt_rows)} rows, {n_outcome} coded outcome=1")

    if n_outcome == 0:
        w = (
            "No truth-table rows pass freq_threshold and consist_threshold — "
            "solutions will be empty.  Consider lowering consist_threshold."
        )
        _log("warn", w)
        warnings_out.append(w)

    # ── Step 3: Boolean minimization (3 solutions) ────────────────────────────
    solutions: list = []
    for sol_type in ("complex", "parsimonious", "intermediate"):
        _log("step", f"Minimizing: {sol_type} solution")
        if sol_type == "intermediate":
            sol = minimize_boolean(
                tt_rows,
                sol_type,
                fuzzy_df,
                list(condition_cols),
                outcome_col,
                directional_expectations=directional_expectations,
            )
        else:
            sol = minimize_boolean(
                tt_rows,
                sol_type,
                fuzzy_df,
                list(condition_cols),
                outcome_col,
            )
        solutions.append(sol)

    # ── Step 4: Coincidence / bubble-chart data ───────────────────────────────
    _log("step", "Generating coincidence plot data")
    bubble_data = generate_coincidence_data(fuzzy_df, outcome_col, list(condition_cols))

    _log("ok", f"fsQCA complete — {n_obs} obs, {len(solutions)} solution(s)")

    # ── Extract UI shortcut fields from the complex solution ──────────────
    # Tests and the JS _renderFsQCAResults() function read these top-level
    # keys directly; the full three-solution breakdown remains in `solutions`.
    complex_sol = next((s for s in solutions if s.solution_type == "complex"), None)
    minimized_solution_str: Optional[str] = None
    consistency_val: Optional[float] = None
    coverage_val: Optional[float] = None
    if complex_sol is not None:
        minimized_solution_str = (
            " + ".join(t.configuration for t in complex_sol.terms)
            if complex_sol.terms else ""
        )
        consistency_val = complex_sol.solution_consistency
        coverage_val = complex_sol.solution_coverage

    return FsQCAResult(
        outcome=outcome_col,
        conditions=list(condition_cols),
        n_obs=n_obs,
        necessity=necessity,
        truth_table=tt_rows,
        solutions=solutions,
        minimized_solution=minimized_solution_str,
        consistency=consistency_val,
        coverage=coverage_val,
        warnings=warnings_out,
        bubble_chart_data=bubble_data,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public function 6 — generate_coincidence_data  (Deliverable 6)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_coincidence_data(
    fuzzy_df: pd.DataFrame,
    outcome_col: str,
    condition_cols: list[str],
) -> list:
    """
    Build XY bubble-chart data for a fuzzy-set coincidence plot.

    For every (case i, condition c) pair returns one :class:`BubbleChartPoint`:

        x_membership = calibrated score of case i in condition c
        y_membership = calibrated score of case i in the outcome

    Total points = n_obs × len(condition_cols).  If this exceeds **2 000**,
    cases are randomly sub-sampled (``random_state=42``) to stay within the cap.

    Returns
    -------
    list[BubbleChartPoint]
    """
    from app.schemas import BubbleChartPoint

    n_obs   = len(fuzzy_df)
    n_conds = len(condition_cols)
    cap     = 2_000

    if n_obs * n_conds > cap:
        max_rows = max(1, cap // n_conds)
        rng      = random.Random(42)  # nosec B311 — seeded for reproducible statistical sampling, not security
        idx_pool = rng.sample(range(n_obs), min(max_rows, n_obs))
        row_indices = sorted(idx_pool)
    else:
        row_indices = list(range(n_obs))

    Y = fuzzy_df[outcome_col].values.astype(float)

    points: list = []
    for i in row_indices:
        y_val = round(float(Y[i]), 4)
        for col in condition_cols:
            x_val = round(float(fuzzy_df[col].iloc[i]), 4)
            points.append(
                BubbleChartPoint(
                    case_id=i,
                    condition=col,
                    x_membership=x_val,
                    y_membership=y_val,
                )
            )

    return points
