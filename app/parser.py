"""
Parser utilities:
  - parse_lavaan()  : parse lavaan-style model syntax into structured dict
  - parse_excel()   : read .xlsx/.xls files
  - parse_spss()    : read .sav files via pyreadstat (optional dep)
"""

import re
import io
import pandas as pd
from typing import Dict, List


# ─── Lavaan Syntax Parser ─────────────────────────────────────────────────────

def preprocess_lavaan(raw_model: str) -> str:
    # 1. Strip leading/trailing whitespace
    raw_model = raw_model.strip()
    
    # 2. Remove R-style variable assignments (e.g., big5_model <- ' ... ')
    # Matches the start of the string, variable name, <- or =, and an opening quote
    raw_model = re.sub(r"^[a-zA-Z0-9_.]+\s*(?:<-|=)\s*['\"]", "", raw_model)
    
    # 3. Remove the trailing quote if it was wrapped in one
    if raw_model.endswith("'") or raw_model.endswith('"'):
        raw_model = raw_model[:-1]

    # Continue with existing logic
    lines = raw_model.split('\n')
    cleaned_lines = []
    current_line = ""

    for line in lines:
        stripped = line.strip()
        
        # Preserve comments and blank lines
        if not stripped or stripped.startswith('#'):
            if current_line:
                cleaned_lines.append(current_line)
                current_line = ""
            cleaned_lines.append(line)
            continue

        # Build the continuous equation
        if current_line:
            current_line += " " + stripped
        else:
            current_line = stripped

        # If the line ends with a continuation operator, keep reading
        # Otherwise, the equation is finished
        if not (current_line.endswith('+') or current_line.endswith('=~') or current_line.endswith('~~') or current_line.endswith('~')):
            cleaned_lines.append(current_line)
            current_line = ""

    # Catch any leftover string at the very end
    if current_line:
        cleaned_lines.append(current_line)

    return "\n".join(cleaned_lines)

def parse_lavaan(model: str) -> Dict:
    """
    Parse lavaan-style syntax into structured components.

    Supported operators:
      =~   measurement (LV =~ indicators)
      ~    structural / regression
      ~~   covariance (optional, future)

    Returns:
      {
        "measurement": {"Trust": ["t1", "t2", "t3"], ...},
        "structural":  [{"lhs": "Satisfaction", "rhs": "Trust"}, ...],
        "covariances": [...],
        "latent_vars": ["Trust", "Satisfaction"],
        "observed_vars": ["t1", "t2", ...],
      }
    """
    model = preprocess_lavaan(model)
    lines = []
    for raw in model.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)

    if not lines:
        raise ValueError("Model syntax is empty.")

    measurement: Dict[str, List[str]] = {}
    structural: List[Dict] = []
    covariances: List[Dict] = []

    for line in lines:
        if "=~" in line:
            parts = line.split("=~", 1)
            lv = parts[0].strip()
            indicators = [v.strip() for v in re.split(r"\+", parts[1]) if v.strip()]
            if not lv:
                raise ValueError(f"Missing LHS in: {line}")
            if not indicators:
                raise ValueError(f"No indicators found in: {line}")
            if lv in measurement:
                measurement[lv].extend(indicators)
            else:
                measurement[lv] = indicators

        elif "~~" in line:
            parts = line.split("~~", 1)
            covariances.append({"lhs": parts[0].strip(), "rhs": parts[1].strip()})

        elif "~" in line:
            parts = line.split("~", 1)
            lhs = parts[0].strip()
            rhs_vars = [v.strip() for v in re.split(r"\+", parts[1]) if v.strip()]
            if not lhs:
                raise ValueError(f"Missing LHS in structural path: {line}")
            if not rhs_vars:
                raise ValueError(f"No predictors found in structural path: {line}")
            for rhs in rhs_vars:
                structural.append({"lhs": lhs, "rhs": rhs})

    latent_vars = list(measurement.keys())
    observed_vars = list({v for inds in measurement.values() for v in inds})

    return {
        "measurement": measurement,
        "structural": structural,
        "covariances": covariances,
        "latent_vars": latent_vars,
        "observed_vars": observed_vars,
    }


def build_semopy_syntax(parsed: Dict) -> str:
    """Convert parsed dict back to semopy-compatible lavaan string."""
    lines = []
    for lv, indicators in parsed["measurement"].items():
        lines.append(f"{lv} =~ {' + '.join(indicators)}")
    for rel in parsed["structural"]:
        lines.append(f"{rel['lhs']} ~ {rel['rhs']}")
    for cov in parsed["covariances"]:
        lines.append(f"{cov['lhs']} ~~ {cov['rhs']}")
    return "\n".join(lines)


# ─── File Parsers ─────────────────────────────────────────────────────────────

def parse_excel(content: bytes) -> pd.DataFrame:
    """Read .xlsx or .xls from raw bytes."""
    try:
        df = pd.read_excel(io.BytesIO(content))
        return df
    except Exception as e:
        raise ValueError(f"Excel parse error: {e}")


def parse_spss(content: bytes) -> pd.DataFrame:
    """
    Read .sav SPSS file from raw bytes.
    Requires: pip install pyreadstat
    """
    try:
        import pyreadstat
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            df, meta = pyreadstat.read_sav(tmp_path)
        finally:
            os.unlink(tmp_path)
        return df
    except ImportError:
        raise ValueError(
            "pyreadstat is required for SPSS files. "
            "Install it with: pip install pyreadstat"
        )
    except Exception as e:
        raise ValueError(f"SPSS parse error: {e}")


def parse_csv_robust(content: bytes) -> pd.DataFrame:
    """
    Robust CSV/TSV parser.

    Tries:
      1. Normal comma CSV
      2. Auto delimiter sniffing
      3. Tab-separated fallback
      4. Semicolon-separated fallback

    Also strips BOMs and weird whitespace.
    """

    import io
    import pandas as pd

    attempts = []

    parsers = [
        {"sep": ","},
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ";"},
    ]

    for opts in parsers:
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                encoding="utf-8-sig",
                **opts
            )

            # reject fake one-column parses
            if len(df.columns) == 1:
                col = str(df.columns[0])

                suspicious = (
                    "\t" in col or
                    ";" in col or
                    "," in col
                )

                if suspicious:
                    raise ValueError(
                        f"Likely wrong delimiter parse: {col[:100]}"
                    )

            # normalize column names
            df.columns = (
                df.columns
                .astype(str)
                .str.strip()
            )

            return df

        except Exception as e:
            attempts.append(str(e))

    raise ValueError(
        "Could not parse CSV/TSV file. "
        f"Tried multiple strategies. Errors: {attempts}"
    )

# ─── Higher-Order Construct helpers ───────────────────────────────────────────

def detect_hoc(parsed: dict) -> dict:
    """
    Detect Higher-Order Constructs (HOCs) in a parsed model.

    A HOC is any latent variable whose measurement block contains the *name of
    another latent variable* as one of its indicators.  The return value maps
    each HOC to the list of First-Order Constructs (FOCs) that act as its
    indicators:

        { "HOC_LV": ["FOC1", "FOC2", ...], ... }

    An empty dict means no HOCs were found.

    Example lavaan syntax that triggers detection::

        HOC  =~ FOC1 + FOC2
        FOC1 =~ x1 + x2 + x3
        FOC2 =~ x4 + x5 + x6
    """
    lv_set = set(parsed.get("latent_vars", []))
    measurement = parsed.get("measurement", {})
    hoc_map: dict[str, list[str]] = {}
    for lv, indicators in measurement.items():
        focs = [ind for ind in indicators if ind in lv_set]
        if focs:
            hoc_map[lv] = focs
    return hoc_map


def expand_hoc_repeated_indicator(parsed: dict) -> dict:
    """
    Transform a parsed model for the **repeated indicator** approach.

    For every HOC whose measurement block contains FOC names, replace those
    FOC names with the *union of all that FOC's own indicators*.  The FOC
    measurement blocks are left intact so both levels are estimated jointly.

    Example transformation::

        Before:
            HOC  =~ FOC1 + FOC2
            FOC1 =~ x1 + x2
            FOC2 =~ x3 + x4

        After:
            HOC  =~ x1 + x2 + x3 + x4   ← FOC indicators copied up
            FOC1 =~ x1 + x2
            FOC2 =~ x3 + x4

    Returns a deep-copy of ``parsed`` with the HOC blocks expanded.
    Raises ValueError if the expansion would produce an empty indicator list.
    """
    import copy
    result = copy.deepcopy(parsed)
    hoc_map = detect_hoc(parsed)
    if not hoc_map:
        return result

    measurement = result["measurement"]
    for hoc, focs in hoc_map.items():
        expanded: list[str] = []
        for ind in measurement[hoc]:
            if ind in measurement:          # it's a FOC name — expand it
                expanded.extend(measurement[ind])
            else:
                expanded.append(ind)        # it's already an observed indicator
        if not expanded:
            raise ValueError(
                f"HOC repeated-indicator expansion produced no indicators for '{hoc}'."
            )
        measurement[hoc] = expanded

    # Recompute observed_vars from the new measurement blocks
    result["observed_vars"] = list({
        v for inds in measurement.values() for v in inds
    })
    return result


def build_hoc_stage2_parsed(parsed: dict, stage1_score_cols: dict[str, str]) -> dict:
    """
    Build a Stage-2 parsed dict for the **two-stage** HOC approach.

    Parameters
    ----------
    parsed : dict
        Original ``parse_lavaan()`` output containing HOC definitions.
    stage1_score_cols : dict
        Mapping ``{foc_name: score_column_name}`` for every FOC whose scores
        were extracted in Stage 1 (e.g. ``{"FOC1": "__score_FOC1__", ...}``).

    Returns a new ``parsed`` dict where:
    - Each HOC's measurement block uses the score column names instead of
      the FOC LV names.
    - FOC measurement blocks are removed (they are now observed variables).
    - Structural paths are updated so any reference to a FOC that now has a
      score column is renamed to that score column name.
    - ``latent_vars`` / ``observed_vars`` are recomputed consistently.

    Raises ValueError if no HOCs are found.
    """
    import copy
    hoc_map = detect_hoc(parsed)
    if not hoc_map:
        raise ValueError("build_hoc_stage2_parsed: no HOCs found in model.")

    measurement_orig = parsed.get("measurement", {})
    foc_set = {foc for focs in hoc_map.values() for foc in focs}

    # Rename helper
    def _rn(name: str) -> str:
        return stage1_score_cols.get(name, name)

    # Stage-2 measurement: keep non-FOC blocks; swap FOC names in HOC blocks
    new_measurement: dict[str, list[str]] = {}
    for lv, inds in measurement_orig.items():
        if lv in foc_set:
            continue                        # drop — their scores are observed vars now
        new_indicators = [_rn(ind) for ind in inds]
        new_measurement[lv] = new_indicators

    # Stage-2 structural: rename FOC LV refs to score column names
    new_structural: list[dict] = []
    for rel in parsed.get("structural", []):
        lhs = _rn(rel["lhs"])
        rhs = _rn(rel["rhs"])
        new_structural.append({"lhs": lhs, "rhs": rhs})

    new_covariances: list[dict] = []
    for cov in parsed.get("covariances", []):
        new_covariances.append({"lhs": _rn(cov["lhs"]), "rhs": _rn(cov["rhs"])})

    new_lv = list(new_measurement.keys())
    new_obs = list({v for inds in new_measurement.values() for v in inds})

    return {
        "measurement": new_measurement,
        "structural":  new_structural,
        "covariances": new_covariances,
        "latent_vars": new_lv,
        "observed_vars": new_obs,
    }


# ─── v0.7: Interaction / Moderation helpers ───────────────────────────────────

def detect_interactions(parsed: dict) -> list:
    """
    Scan the structural block for interaction terms (``X*M`` notation).

    Returns a list of dicts, one per detected interaction::

        [{"iv": "X", "moderator": "M", "term": "X*M",
          "outcome": "Y", "interaction_col": "X_x_M"}, ...]

    ``interaction_col`` is the sanitised column name that will be created in
    the dataframe by :func:`expand_interaction_terms`.

    A term is recognised when a structural RHS variable contains ``*``.
    Example input lavaan syntax::

        Y ~ X + M + X*M
        X =~ x1 + x2
        M =~ m1 + m2
    """
    interactions = []
    seen = set()

    for rel in parsed.get("structural", []):
        rhs = rel["rhs"]
        if "*" not in rhs:
            continue

        parts = [p.strip() for p in rhs.split("*", 1)]
        if len(parts) != 2:
            continue

        iv, mod = parts[0], parts[1]
        key = (iv, mod, rel["lhs"])
        if key in seen:
            continue
        seen.add(key)

        # Sanitise: replace special chars with underscore for column name
        col = re.sub(r"[^A-Za-z0-9_]", "_", f"{iv}_x_{mod}")
        interactions.append({
            "iv":              iv,
            "moderator":       mod,
            "term":            rhs,
            "outcome":         rel["lhs"],
            "interaction_col": col,
        })

    return interactions


def expand_interaction_terms(
    parsed: dict,
    df: "pd.DataFrame",
) -> "tuple[dict, pd.DataFrame]":
    """
    Create mean-centred product columns for all interaction terms and update
    the parsed model dict so the structural block references the new column
    names instead of the ``X*M`` notation.

    Procedure
    ---------
    For each interaction ``X*M → Y``:

    1. Compute composite score for X and M (mean of their indicators, or
       the column itself if X / M is an observed variable).
    2. Mean-centre both composites.
    3. Create the product column: ``df["X_x_M"] = X_mc × M_mc``.
    4. In the parsed structural block, replace ``{"lhs": Y, "rhs": "X*M"}``
       with ``{"lhs": Y, "rhs": "X_x_M"}``.
    5. The interaction column is treated as a single-indicator observed
       variable — it does NOT get its own ``=~`` block.

    Returns
    -------
    (new_parsed, df_augmented)
        ``new_parsed`` has the ``X*M`` rhs entries replaced and an extra key
        ``"interactions"`` that mirrors :func:`detect_interactions` output.
        ``df_augmented`` contains all original columns plus the product
        column(s).

    Raises
    ------
    ValueError
        If a composite cannot be constructed for X or M.
    """
    import copy
    import numpy as np

    interactions = detect_interactions(parsed)
    if not interactions:
        return parsed, df.copy()

    result   = copy.deepcopy(parsed)
    df_aug   = df.copy()
    meas     = result.get("measurement", {})

    def _composite(name: str) -> "np.ndarray":
        """Return mean composite for LV *name* or raw column if observed."""
        if name in meas:
            cols = [c for c in meas[name] if c in df_aug.columns]
            if not cols:
                raise ValueError(
                    f"expand_interaction_terms: no indicator columns found "
                    f"for LV '{name}'. Check that the dataset is uploaded."
                )
            return df_aug[cols].astype(float).mean(axis=1).values
        elif name in df_aug.columns:
            return df_aug[name].astype(float).values
        else:
            raise ValueError(
                f"expand_interaction_terms: '{name}' is neither a latent "
                f"variable in the model nor a column in the dataset."
            )

    for itx in interactions:
        iv_comp  = _composite(itx["iv"])
        mod_comp = _composite(itx["moderator"])

        # Mean-centre
        iv_mc  = iv_comp  - iv_comp.mean()
        mod_mc = mod_comp - mod_comp.mean()

        col = itx["interaction_col"]
        df_aug[col] = iv_mc * mod_mc

        # Patch structural block: replace X*M → interaction_col
        new_structural = []
        for rel in result["structural"]:
            if rel["rhs"] == itx["term"] and rel["lhs"] == itx["outcome"]:
                new_structural.append({"lhs": rel["lhs"], "rhs": col})
            else:
                new_structural.append(rel)
        result["structural"] = new_structural

        # The interaction column is observed — add to observed_vars if absent
        obs = result.get("observed_vars", [])
        if col not in obs:
            obs.append(col)
        result["observed_vars"] = obs

    result["interactions"] = interactions
    return result, df_aug
