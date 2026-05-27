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
