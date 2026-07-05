"""
engine_utils_additions.py — v1.1 Feature A17: plain-English annotations
=========================================================================
MERGE TARGET: app/engine_utils.py (not uploaded this session — see the
summary at the end of the chat for why this ships as a separate file
instead of a direct edit).

Everything below is self-contained (stdlib only) and can be pasted
directly into engine_utils.py. If engine_utils.py already defines an
APA-style number formatter (export_docx.py and export_pdf.py each have
their own _fmt/_fmt_p — engine_utils.py may already have an equivalent),
reuse that instead of the local _apa_num/_apa_p below to avoid two
sources of truth for number formatting.

Public API (as specified in the ticket)
----------------------------------------
  annotate_path_coefficient(estimate, se, p_value)      -> str
  annotate_fit_index(name, value)                        -> str
  annotate_indirect_effect(estimate, ci_lower, ci_upper) -> str

Each returns ONE auto-generated sentence, Amos-"use-it-in-a-sentence"
style. These are additive — callers keep reporting the numeric fields
as before and append the sentence to an `annotations: list[str]` field
(already added to ModelResult, IndirectResult, BootstrapResult, and
NCAResult in schemas.py this session).
"""

from __future__ import annotations

import re
from typing import Optional


# ── APA-style number formatting ──────────────────────────────────────────────
# APA convention: statistics that cannot exceed 1 in absolute value
# (standardised coefficients, correlations, p-values) are reported without
# a leading zero, e.g. ".14" not "0.14". export_docx.py / export_pdf.py
# already follow this for tables (_fmt_p); annotate_* follows it for prose.

def _apa_num(value: Optional[float], dp: int = 2) -> str:
    """Format *value* to *dp* decimals, APA-style (no leading zero)."""
    if value is None:
        return "n/a"
    s = f"{value:.{dp}f}"
    if value != 0:
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    return s


def _apa_p(p: Optional[float]) -> str:
    """APA-style p-value text: '< .001', else e.g. '.012' (3 dp)."""
    if p is None:
        return "n/a"
    if p < 0.001:
        return "< .001"
    return _apa_num(p, 3)


# ── annotate_path_coefficient ────────────────────────────────────────────────

def annotate_path_coefficient(
    estimate: float,
    se: Optional[float] = None,
    p_value: Optional[float] = None,
) -> str:
    """
    One sentence describing a structural or measurement path coefficient.

    Magnitude bands are a rough heuristic for STANDARDISED path
    coefficients (loosely after Chin, 1998 / Lohm\u00f6ller, 1989 — swap in
    your field's convention if you follow a stricter rule):
        |\u03b2| < .10          negligible
        .10 \u2264 |\u03b2| < .20   small
        .20 \u2264 |\u03b2| < .35   moderate
        |\u03b2| \u2265 .35          substantial

    Significance uses the conventional \u03b1 = .05 cutoff.

    Examples
    --------
    >>> annotate_path_coefficient(0.42, se=0.05, p_value=0.0001)
    'This path coefficient (\u03b2 = .42, SE = .05) reflects a substantial
    positive effect and is statistically significant (p < .001).'
    >>> annotate_path_coefficient(-0.07, se=0.09, p_value=0.44)
    'This path coefficient (\u03b2 = -.07, SE = .09) reflects a negligible
    negative effect and is not statistically significant (p = .440).'
    """
    direction = "positive" if estimate >= 0 else "negative"
    abs_est = abs(estimate)
    if abs_est < 0.10:
        magnitude = "negligible"
    elif abs_est < 0.20:
        magnitude = "small"
    elif abs_est < 0.35:
        magnitude = "moderate"
    else:
        magnitude = "substantial"

    detail = f"\u03b2 = {_apa_num(estimate, 2)}"
    if se is not None:
        detail += f", SE = {_apa_num(se, 2)}"

    if p_value is None:
        return (
            f"This path coefficient ({detail}) reflects a {magnitude} "
            f"{direction} effect; its statistical significance could not "
            "be determined because no p-value was available."
        )

    p_text = _apa_p(p_value)
    p_clause = f"p {p_text}" if p_text.startswith("<") else f"p = {p_text}"
    sig_clause = (
        f"statistically significant ({p_clause})" if p_value < 0.05
        else f"not statistically significant ({p_clause})"
    )
    return (
        f"This path coefficient ({detail}) reflects a {magnitude} "
        f"{direction} effect and is {sig_clause}."
    )


# ── annotate_indirect_effect ─────────────────────────────────────────────────

def annotate_indirect_effect(
    estimate: float,
    ci_lower: Optional[float],
    ci_upper: Optional[float],
) -> str:
    """
    One sentence describing a bootstrapped indirect (mediated) effect.

    Significance is determined by whether the bootstrap CI excludes zero
    (the standard PLS-SEM mediation-testing convention — Hair et al., 2022,
    Ch. 7 — not a p-value / normality-based test).

    Example (matches the ticket's sample sentence exactly)
    --------------------------------------------------------
    >>> annotate_indirect_effect(0.14, 0.06, 0.23)
    'This indirect effect (b = .14) is significant: the 95% bootstrap CI
    [.06, .23] excludes zero.'
    """
    if ci_lower is None or ci_upper is None:
        return (
            f"This indirect effect (b = {_apa_num(estimate, 2)}) could not "
            "be tested for significance because a bootstrap confidence "
            "interval is not available."
        )

    excludes_zero = ci_lower > 0 or ci_upper < 0
    verdict = "significant" if excludes_zero else "not significant"
    relation = "excludes" if excludes_zero else "includes"

    return (
        f"This indirect effect (b = {_apa_num(estimate, 2)}) is {verdict}: "
        f"the 95% bootstrap CI [{_apa_num(ci_lower, 2)}, "
        f"{_apa_num(ci_upper, 2)}] {relation} zero."
    )


# ── annotate_fit_index ───────────────────────────────────────────────────────
#
# Threshold table. CFI / TLI / RMSEA / SRMR bands are copied from the
# *_acceptable / *_good flags already computed in FitIndices (schemas.py)
# / engine._fit_verdict (v1.1 A7) — keep both in sync if either changes,
# ideally by having one of them import the cutoffs from the other instead
# of hardcoding them twice.
#
# `bands` is ordered STRICTEST first: for "higher_better" metrics, highest
# threshold first; for "lower_better" metrics, lowest threshold first.
# `below_label` / its implied threshold (the last band) covers values that
# don't clear any band.

_FIT_INDEX_RULES: dict = {
    "cfi": dict(label="CFI", direction="higher_better",
                citation="Hu & Bentler, 1999",
                bands=[(0.95, "good"), (0.90, "acceptable")],
                below_label="below conventional fit standards"),
    "tli": dict(label="TLI", direction="higher_better",
                citation="Hu & Bentler, 1999",
                bands=[(0.95, "good"), (0.90, "acceptable")],
                below_label="below conventional fit standards"),
    "rmsea": dict(label="RMSEA", direction="lower_better",
                  citation="Hu & Bentler, 1999; MacCallum et al., 1996",
                  bands=[(0.06, "good"), (0.08, "acceptable")],
                  below_label="above conventional fit standards"),
    "srmr": dict(label="SRMR", direction="lower_better",
                 citation="Hu & Bentler, 1999; Henseler et al., 2014",
                 bands=[(0.08, "good")],
                 below_label="above the conventional cutoff"),
    "gfi": dict(label="GFI", direction="higher_better",
                citation="J\u00f6reskog & S\u00f6rbom, 1984",
                bands=[(0.90, "acceptable")],
                below_label="below the conventional cutoff"),
    "agfi": dict(label="AGFI", direction="higher_better",
                 citation="J\u00f6reskog & S\u00f6rbom, 1984",
                 bands=[(0.90, "acceptable")],
                 below_label="below the conventional cutoff"),
    "nfi": dict(label="NFI", direction="higher_better",
                citation="Bentler & Bonett, 1980",
                bands=[(0.90, "acceptable")],
                below_label="below the conventional cutoff"),
    "r2": dict(label="R\u00b2", direction="higher_better",
               citation="Hair et al., 2022",
               bands=[(0.75, "substantial"), (0.50, "moderate"), (0.25, "weak")],
               below_label="very weak"),
    "q2": dict(label="Q\u00b2", direction="higher_better",
               citation="Geisser, 1974; Stone, 1974",
               bands=[(0.0, "predictively relevant")],
               below_label="lacking predictive relevance"),
    "f2": dict(label="f\u00b2", direction="higher_better",
               citation="Cohen, 1988",
               bands=[(0.35, "large"), (0.15, "medium"), (0.02, "small")],
               below_label="negligible"),
    "ave": dict(label="AVE", direction="higher_better",
                citation="Fornell & Larcker, 1981",
                bands=[(0.50, "adequate")],
                below_label="below the convergent-validity threshold"),
    "vif": dict(label="VIF", direction="lower_better",
                citation="Hair et al., 2022",
                bands=[(3.30, "conservatively acceptable"), (5.00, "acceptable")],
                below_label="a multicollinearity concern"),
    "htmt": dict(label="HTMT", direction="lower_better",
                 citation="Henseler, Ringle & Sarstedt, 2015",
                 bands=[(0.90, "acceptable")],
                 below_label="a discriminant-validity concern"),
    # NCA ceiling-line effect sizes (Dul, 2016) — distinct thresholds from
    # Cohen's f2 above, so these are NOT aliased to "f2".
    "ce_fdh_d": dict(label="CE-FDH effect size (d)", direction="higher_better",
                      citation="Dul, 2016",
                      bands=[(0.50, "large"), (0.30, "medium"), (0.10, "small")],
                      below_label="negligible"),
    "cr_fdh_d": dict(label="CR-FDH effect size (d)", direction="higher_better",
                      citation="Dul, 2016",
                      bands=[(0.50, "large"), (0.30, "medium"), (0.10, "small")],
                      below_label="negligible"),
}

_NAME_ALIASES = {
    "rsquared": "r2", "r2": "r2",
    "nnfi": "tli", "tli": "tli",
    "cfi": "cfi", "rmsea": "rmsea", "srmr": "srmr",
    "gfi": "gfi", "agfi": "agfi", "nfi": "nfi",
    "q2": "q2", "qsquared": "q2",
    "f2": "f2", "fsquared": "f2", "cohensf2": "f2",
    "ave": "ave", "vif": "vif", "htmt": "htmt",
    "cefdhd": "ce_fdh_d", "cefdh": "ce_fdh_d",
    "crfdhd": "cr_fdh_d", "crfdh": "cr_fdh_d",
}


def _normalize_index_name(name: str) -> str:
    """'R\u00b2' / 'R-squared' / 'r_squared' / ' R2 ' all normalise to 'r2'."""
    n = name.strip().lower()
    n = n.replace("\u00b2", "2").replace("\u00b9", "1")
    n = re.sub(r"[\s\-_]+", "", n)
    return _NAME_ALIASES.get(n, n)


def _evaluate_band(value: float, bands: list, direction: str):
    """Return (label, threshold) for the first band satisfied, else (None, None)."""
    for threshold, label in bands:
        if direction == "higher_better" and value >= threshold:
            return label, threshold
        if direction == "lower_better" and value <= threshold:
            return label, threshold
    return None, None


def annotate_fit_index(name: str, value: Optional[float]) -> str:
    """
    One sentence interpreting a single named fit index or effect-size
    statistic.

    Recognised names (case/spacing/underscore-insensitive; \u00b2 normalises
    to '2'): CFI, TLI/NNFI, RMSEA, SRMR, GFI, AGFI, NFI, R2/R-squared, Q2,
    f2, AVE, VIF, HTMT, CE-FDH d, CR-FDH d.

    Unrecognised names get a neutral, non-evaluative sentence rather than
    a guessed threshold — extend _FIT_INDEX_RULES to add one.

    Example
    -------
    >>> annotate_fit_index("SRMR", 0.061)
    'SRMR = .061, which is good (\u2264 .08; Hu & Bentler, 1999; Henseler et
    al., 2014).'
    """
    if value is None:
        return f"{name} is not available."

    rule = _FIT_INDEX_RULES.get(_normalize_index_name(name))
    if rule is None:
        return (
            f"{name} = {_apa_num(value, 3)}. (No interpretation rule is "
            "registered for this index — add one to _FIT_INDEX_RULES if "
            "it should be evaluated.)"
        )

    label, threshold = _evaluate_band(value, rule["bands"], rule["direction"])
    if label is None:
        label = rule["below_label"]
        threshold = rule["bands"][-1][0]
    symbol = "\u2265" if rule["direction"] == "higher_better" else "\u2264"

    return (
        f"{rule['label']} = {_apa_num(value, 3)}, which is {label} "
        f"({symbol} {_apa_num(threshold, 2)}; {rule['citation']})."
    )


if __name__ == "__main__":
    # Lightweight self-test — not a substitute for real unit tests once
    # merged, but confirms the functions run and match the ticket's example.
    s1 = annotate_indirect_effect(0.14, 0.06, 0.23)
    print(s1)
    _expected_s1 = (
        "This indirect effect (b = .14) is significant: the 95% bootstrap "
        "CI [.06, .23] excludes zero."
    )
    if s1 != _expected_s1:
        raise ValueError(
            f"Self-test failed: does not match the ticket's example "
            f"sentence.\n  got:      {s1!r}\n  expected: {_expected_s1!r}"
        )

    print(annotate_indirect_effect(0.03, -0.02, 0.09))
    print(annotate_indirect_effect(0.14, None, None))
    print()
    print(annotate_path_coefficient(0.42, se=0.05, p_value=0.0001))
    print(annotate_path_coefficient(-0.07, se=0.09, p_value=0.44))
    print(annotate_path_coefficient(0.25, se=None, p_value=None))
    print()
    for nm, val in [("CFI", 0.97), ("CFI", 0.92), ("CFI", 0.80),
                     ("RMSEA", 0.04), ("RMSEA", 0.07), ("RMSEA", 0.15),
                     ("SRMR", 0.061), ("R2", 0.61), ("r_squared", 0.10),
                     ("Q2", -0.02), ("f2", 0.18), ("AVE", 0.44),
                     ("VIF", 2.1), ("VIF", 8.4), ("HTMT", 0.95),
                     ("CE-FDH d", 0.38), ("made_up_index", 1.23)]:
        print(annotate_fit_index(nm, val))

    print("\nAll self-checks passed.")
