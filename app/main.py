"""
NAVAL-SEM — FastAPI Backend + Static Frontend Server
Run standalone: uvicorn app.main:app --reload --port 8000
Run via launcher: python launcher.py
"""

import os
import io
import uuid
import json
import time
import asyncio
import hashlib
import sys
import platform
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

# ── Single source of truth for the app version ───────────────────────────────
APP_VERSION = "0.7.0"
_GITHUB_REPO  = "navalsingh9/naval-sem"


# ── Per-run log store ─────────────────────────────────────────────────────────
# Keyed by run_id; entries are {"t": ms_epoch, "level": str, "msg": str}
# "done" is set True once the run completes so the SSE stream can terminate.
_run_store: dict[str, dict] = {}


def _init_run(run_id: str):
    _run_store[run_id] = {"logs": [], "done": False, "fingerprint": None, "audit": None}
    # Prune old runs (keep last 200) — only evict completed runs to avoid
    # deleting an in-flight run_id and causing KeyError when it tries to mark done.
    if len(_run_store) > 200:
        completed = [k for k in _run_store if k != run_id and _run_store[k]["done"]]
        oldest = sorted(completed, key=lambda k: _run_store[k]["logs"][0]["t"]
                        if _run_store[k]["logs"] else 0)[:max(0, len(_run_store) - 200)]
        for k in oldest:
            del _run_store[k]


def _make_log_fn(run_id: str):
    """Return a (level, msg) → None callback that appends to _run_store."""
    def log_fn(level: str, msg: str):
        if run_id in _run_store:
            _run_store[run_id]["logs"].append({
                "t": round(time.time() * 1000),
                "level": level.lower(),
                "msg": msg,
            })
    return log_fn


def _compute_fingerprint(
    run_id: str,
    model_syntax: str,
    df: pd.DataFrame,
    algorithm: str,
    result,
) -> tuple[str, dict]:
    """
    Compute a SHA-256 fingerprint of the run for reproducibility anchoring.
    The fingerprint covers: model syntax, data hash, algorithm, env, key results.
    Nothing sensitive (raw data) is included — only hashes and aggregates.
    """
    try:
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df.sort_index(axis=1), index=True)
            .values.tobytes()
        ).hexdigest()
    except Exception:
        data_hash = hashlib.sha256(str(df.shape).encode()).hexdigest()

    env = {
        "naval_sem_version": APP_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "numpy": getattr(__import__("numpy"), "__version__", "?"),
        "pandas": getattr(__import__("pandas"), "__version__", "?"),
    }
    try:
        import semopy
        env["semopy"] = getattr(semopy, "__version__", "?")
    except Exception:
        env["semopy"] = "unknown"

    results_summary = {
        "n_obs": result.n_obs,
        "n_params": result.n_params,
        "converged": result.converged,
        "algorithm": result.algorithm,
        "cfi": result.fit.cfi,
        "rmsea": result.fit.rmsea,
        "srmr": result.fit.srmr,
        "aic": result.fit.aic,
        "bic": result.fit.bic,
    }

    payload = {
        "run_id": run_id,
        "model_syntax": model_syntax.strip(),
        "data_hash": data_hash,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "algorithm": algorithm,
        "results": results_summary,
        "env": env,
    }

    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    audit = {**payload, "fingerprint": fingerprint, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return fingerprint, audit

from app.engine import (
    fit_model, run_bootstrap, compute_htmt, export_as_code,
    compute_indirect_effects, compute_cmb, compute_predict,
)
from app.parser import (parse_spss, parse_excel, parse_lavaan, parse_csv_robust,)
from app.schemas import (
    ModelResult, BootstrapResult, HTMTResult, IndirectResult,
    CMBMarkerResult, PredictResult, MGAResult,
    ModerationResult, IPMAResult, NCAResult,          # v0.7
    ModMediationResult,                               # v0.7 — moderated mediation
)
from app.engine_mga import run_mga, fit_hoc_repeated_indicator, fit_hoc_two_stage
from app.engine_moderation    import run_moderation                                # v0.7
from app.engine_ipma          import compute_ipma                                  # v0.7
from app.nca                  import compute_nca                                   # v0.7
from app.engine_mod_mediation import run_mod_mediation                             # v0.7


def _manifest_moderation(
    df: pd.DataFrame,
    model_syntax: str,
    bootstrap_n: int = 0,
    log_fn=None,
) -> dict:
    """
    OLS fallback for moderation models that have no latent variables (no =~ lines).
    Handles ``Y ~ X + M + X*M`` syntax for observed variables only.
    Returns a dict shaped like ModerationResult so FastAPI can serialize it.
    """
    import re
    import numpy as np

    if log_fn:
        log_fn("info", "No latent variables detected — using OLS fallback for manifest moderation.")

    moderation_terms: list = []
    all_params: list = []

    for raw_line in model_syntax.strip().split("\n"):
        line = raw_line.strip()
        if not line or "=~" in line or "~~" in line or "~" not in line:
            continue

        lhs, rhs = line.split("~", 1)
        lhs = lhs.strip()
        if lhs not in df.columns:
            continue

        rhs_parts = [p.strip() for p in rhs.split("+")]
        interactions = [p for p in rhs_parts if "*" in p]
        plain_preds  = [p for p in rhs_parts if "*" not in p and p in df.columns]

        y = df[lhs].values.astype(float)
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        if not interactions:
            # Plain OLS regression — emit parameters only
            if not plain_preds:
                continue
            X = np.column_stack([np.ones(len(y))] + [df[c].values.astype(float) for c in plain_preds])
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            for i, col in enumerate(plain_preds):
                all_params.append({
                    "lhs": lhs, "op": "~", "rhs": col,
                    "estimate": round(float(coeffs[i + 1]), 6),
                    "std_estimate": None, "std_error": None,
                    "z_value": None, "p_value": None,
                    "ci_lower": None, "ci_upper": None, "significant": None,
                })
            continue

        for interaction in interactions:
            parts = [v.strip() for v in interaction.split("*")]
            if len(parts) != 2:
                continue
            iv, moderator = parts
            if iv not in df.columns or moderator not in df.columns:
                continue

            # Standardise IV and moderator for interpretable β and simple slopes
            iv_vals  = (df[iv].values.astype(float))
            mod_vals = (df[moderator].values.astype(float))
            iv_c     = iv_vals  - iv_vals.mean()
            mod_c    = mod_vals - mod_vals.mean()
            iv_s     = iv_c  / (iv_c.std()  or 1.0)
            mod_s    = mod_c / (mod_c.std() or 1.0)
            int_s    = iv_s * mod_s

            # Full model (with interaction)
            Xf = np.column_stack([np.ones(len(y)), iv_s, mod_s, int_s])
            cf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
            yhat_f  = Xf @ cf
            r2_with = float(1.0 - np.sum((y - yhat_f) ** 2) / ss_tot) if ss_tot else 0.0

            # Reduced model (without interaction)
            Xr = np.column_stack([np.ones(len(y)), iv_s, mod_s])
            cr, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            yhat_r  = Xr @ cr
            r2_without = float(1.0 - np.sum((y - yhat_r) ** 2) / ss_tot) if ss_tot else 0.0

            delta_r2 = max(0.0, r2_with - r2_without)
            f2 = delta_r2 / (1.0 - r2_with) if r2_with < 1.0 else 0.0

            # Bootstrap CIs for interaction β
            beta_int = float(cf[3])
            ci_lower_95 = ci_upper_95 = None
            if bootstrap_n > 0:
                rng = np.random.default_rng(42)
                n, boot = len(y), []
                for _ in range(bootstrap_n):
                    idx = rng.integers(0, n, size=n)
                    try:
                        cb, *_ = np.linalg.lstsq(Xf[idx], y[idx], rcond=None)
                        boot.append(float(cb[3]))
                    except (np.linalg.LinAlgError, ValueError):
                        # Singular or rank-deficient matrix on this bootstrap resample — skip sample
                        pass
                if boot:
                    ci_lower_95 = round(float(np.percentile(boot, 2.5)), 6)
                    ci_upper_95 = round(float(np.percentile(boot, 97.5)), 6)

            significant = bool(
                ci_lower_95 is not None and (ci_lower_95 > 0 or ci_upper_95 < 0)
            )

            # Simple slopes at −1 SD / mean / +1 SD of moderator
            mod_sd = float(mod_c.std() or 1.0)
            simple_slopes = []
            for label, level in [("low (−1 SD)", -1.0), ("mean (0)", 0.0), ("high (+1 SD)", 1.0)]:
                slope = round(float(cf[1] + cf[3] * level), 6)
                simple_slopes.append({
                    "moderator_level": label,
                    "moderator_value": round(level * mod_sd, 3),
                    "slope": slope,
                    "ci_lower_95": None, "ci_upper_95": None,
                    "significant": abs(slope) > 0.05,
                })

            moderation_terms.append({
                "iv": iv, "moderator": moderator, "outcome": lhs,
                "interaction_col": f"{iv}_x_{moderator}",
                "beta_iv": round(float(cf[1]), 6),
                "beta_moderator": round(float(cf[2]), 6),
                "beta_interaction": round(beta_int, 6),
                "ci_lower_95": ci_lower_95, "ci_upper_95": ci_upper_95,
                "significant": significant,
                "r2_with": round(r2_with, 4),
                "r2_without": round(r2_without, 4),
                "delta_r2": round(delta_r2, 4),
                "f2_interaction": round(f2, 4),
                "simple_slopes": simple_slopes,
            })

            for col_name, coeff in [
                (iv, cf[1]), (moderator, cf[2]), (f"{iv}_x_{moderator}", cf[3])
            ]:
                all_params.append({
                    "lhs": lhs, "op": "~", "rhs": col_name,
                    "estimate": round(float(coeff), 6),
                    "std_estimate": None, "std_error": None,
                    "z_value": None, "p_value": None,
                    "ci_lower": None, "ci_upper": None, "significant": None,
                })

    return {
        "algorithm": "ols",
        "n_obs": len(df),
        "bootstrap_n": bootstrap_n,
        "moderation_terms": moderation_terms,
        "parameters": all_params,
        "fit": {
            "cfi": None, "tli": None, "rmsea": None,
            "rmsea_ci_lower": None, "rmsea_ci_upper": None,
            "srmr": None, "chi_square": None, "df": None,
            "p_value": None, "aic": None, "bic": None,
            "r_squared": None,
        },
        "warnings": [
            "OLS fallback: model contains no latent variables (=~). "
            "Moderation computed via ordinary least squares on observed variables.",
        ],
    }

def _expand_covariances(model_syntax: str) -> str:
    """
    TC-52: Expand multi-target residual covariance lines into individual pairs.

    semopy requires each covariance to be on its own line.  lavaan allows shorthand
    like ``y2 ~~ y4 + y6`` which the parser may re-emit verbatim — semopy then fails
    with an eigenvalue convergence error because its model-implied covariance matrix
    is mis-specified.

    Transforms:
        y2 ~~ y4 + y6   →   y2 ~~ y4
                             y2 ~~ y6
    Single-target lines and all other syntax are left unchanged.
    """
    import re
    out = []
    for raw in model_syntax.split("\n"):
        line = raw.strip()
        if "~~" in line and "=~" not in line:
            m = re.match(r"^(\w+)\s*~~\s*([^\r\n]{1,500})$", line)
            if m:
                lhs = m.group(1).strip()
                rhs_parts = [p.strip() for p in m.group(2).split("+") if p.strip()]
                if len(rhs_parts) > 1:
                    out.extend(f"{lhs} ~~ {rhs}" for rhs in rhs_parts)
                    continue
        out.append(raw)
    return "\n".join(out)


def auto_reverse_score(df: pd.DataFrame, model_syntax: str, log_fn=None) -> pd.DataFrame:
    """
    Detects 'r'-prefixed variables in the Lavaan syntax. 
    If 'rVAR' is requested but only 'VAR' exists in the DataFrame, 
    it automatically reverse-scores the column.
    """
    try:
        # We use the existing parser to extract exactly what the user is asking for
        parsed = parse_lavaan(model_syntax)
        observed_vars = parsed.get("observed_vars", [])
    except Exception:
        return df  # If syntax is broken, do nothing and let the engine catch it later

    for var in observed_vars:
        if var not in df.columns:
            # Check if it starts with 'r' and the base variable (without 'r') exists
            if var.startswith('r') and var[1:] in df.columns:
                base_var = var[1:]
                
                # Standard reverse scoring formula: (Maximum + Minimum) - Value
                col_max = df[base_var].max()
                col_min = df[base_var].min()
                
                df[var] = (col_max + col_min) - df[base_var]
                
                if log_fn:
                    log_fn("info", f"Auto-reverse scored '{base_var}' into '{var}' (Detected scale: {col_min} to {col_max})")
    
    return df

_STATIC_DIR = os.environ.get(
    "NAVAL_SEM_STATIC",
    str(Path(__file__).parent.parent / "static"),
)

app = FastAPI(title="NAVAL-SEM API", version=APP_VERSION)

# --- ADDED: Catch Pydantic serialization errors (e.g., NaNs slipping through) ---
from fastapi.exceptions import ResponseValidationError
@app.exception_handler(ResponseValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Response serialization error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Backend generated an invalid response (likely NaN estimates). Check server logs for details."}
    )
# --------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

if Path(_STATIC_DIR).exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(Path(_STATIC_DIR) / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def root_missing():
        return JSONResponse({"error": "Static files not found. Check server configuration."}, status_code=500)



@app.get("/logs/{run_id}")
async def stream_logs(run_id: str):
    """
    Server-Sent Events stream for a running analysis.
    The client opens this immediately after submitting /run, passing the same run_id.
    Streams {"t","level","msg"} entries; terminates with {"done":true,"fingerprint":"..."}.
    """
    async def event_gen():
        wait_time = 0.0
        while _run_store.get(run_id) is None and wait_time < 5.0:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        last = 0
        elapsed = 0.0
        while elapsed < 14400:
            run = _run_store.get(run_id)
            if run is None:
                yield f"data: {json.dumps({'level': 'error', 'msg': 'run_id not found'})}\n\n"
                return
            for entry in run["logs"][last:]:
                yield f"data: {json.dumps(entry)}\n\n"
            last = len(run["logs"])
            if run["done"]:
                fp = run.get("fingerprint")
                audit = run.get("audit")
                yield f"data: {json.dumps({'done': True, 'fingerprint': fp, 'audit': audit})}\n\n"
                return
            await asyncio.sleep(0.2)
            elapsed += 0.2
        yield f"data: {json.dumps({'level': 'warn', 'msg': 'Live log closed after 4 h — computation still running in background. Results will appear when complete.'})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/fingerprint/{run_id}")
async def get_fingerprint(run_id: str):
    """Return the full audit record for a completed run."""
    run = _run_store.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if not run["done"]:
        raise HTTPException(425, "Run not yet complete")
    return {"run_id": run_id, "fingerprint": run["fingerprint"], "audit": run["audit"]}


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/check-updates")
async def check_updates():
    """
    Checks GitHub Releases for a version newer than APP_VERSION.
    Returns update_available=True/False (or status='offline' on network error).
    Uses only stdlib — no extra deps, no API key required.
    """
    import json as _json
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    api_url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"

    def _ver_tuple(v: str):
        try:
            return tuple(int(x) for x in v.lstrip("v").split("."))
        except ValueError:
            return (0,)

    try:
        req = Request(
            api_url,
            headers={
                "User-Agent": f"NAVAL-SEM/{APP_VERSION}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urlopen(req, timeout=8) as resp:  # nosec B310 – URL is a compile-time https:// constant, not user-supplied
            data = _json.loads(resp.read())

        latest_tag  = data.get("tag_name", "").lstrip("v")
        release_url = data.get("html_url", f"https://github.com/{_GITHUB_REPO}/releases")
        release_name = data.get("name") or f"v{latest_tag}"

        update_available = _ver_tuple(latest_tag) > _ver_tuple(APP_VERSION)
        return {
            "current_version": APP_VERSION,
            "latest_version":  latest_tag,
            "release_name":    release_name,
            "release_url":     release_url,
            "update_available": update_available,
        }

    except URLError:
        return {"current_version": APP_VERSION, "status": "offline"}
    except Exception as exc:
        logger.warning(f"check-updates failed: {exc}")  # details stay server-side only
        return {"current_version": APP_VERSION, "status": "error"}


@app.post("/predict", response_model=PredictResult)
async def predictive_relevance(
    file: UploadFile = File(...),
    model: str = Form(...),
    omission_distance: int = Form(7),
    k_folds: int = Form(10),
    missing: str = Form("listwise"),
):
    """
    v0.5 predictive relevance suite.
    Returns Q² (blindfolding), PLSpredict (k-fold vs LM), and CVPAT.
    """
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = parse_csv_robust(content)
        elif ext in ("xlsx", "xls"):
            df = parse_excel(content)
        elif ext == "sav":
            df = parse_spss(content)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /predict: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model)
    try:
        return compute_predict(
            df, model,
            omission_distance=omission_distance,
            k_folds=k_folds,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /predict: %s", e, exc_info=True)
        raise HTTPException(500, "Predictive relevance analysis failed. Check server logs.")


@app.post("/cmb", response_model=CMBMarkerResult)
async def cmb_analysis(
    file: UploadFile = File(...),
    model: str = Form(...),
    marker_variable: str = Form(...),
    missing: str = Form("listwise"),
):
    """
    Common Method Bias marker variable analysis (Lindell & Whitney 2001).
    Provide a marker variable theoretically unrelated to your constructs.
    """
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = parse_csv_robust(content)
        elif ext in ("xlsx", "xls"):
            df = parse_excel(content)
        elif ext == "sav":
            df = parse_spss(content)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /cmb: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model)
    try:
        return compute_cmb(df, model, marker_variable=marker_variable)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /cmb: %s", e, exc_info=True)
        raise HTTPException(500, "CMB analysis failed. Check server logs.")


@app.post("/indirect", response_model=IndirectResult)
async def indirect_effects(
    file: UploadFile = File(...),
    model: str = Form(...),
    bootstrap_n: int = Form(500),
    missing: str = Form("listwise"),
):
    """
    Decompose indirect (mediation) effects for all variable pairs connected
    via paths of length ≥ 2. Returns indirect effects with bootstrapped 95% CIs
    and a total effects matrix (direct + indirect).
    """
    bootstrap_n = min(bootstrap_n, 20_000)
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = parse_csv_robust(content)
        elif ext in ("xlsx", "xls"):
            df = parse_excel(content)
        elif ext == "sav":
            df = parse_spss(content)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /indirect: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model)
    try:
        return compute_indirect_effects(df, model, n_bootstrap=bootstrap_n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /indirect: %s", e, exc_info=True)
        raise HTTPException(500, "Indirect effects computation failed. Check server logs.")


@app.post("/upload/preview")
async def upload_preview(file: UploadFile = File(...)):
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = parse_csv_robust(content)
        elif ext in ("xlsx", "xls"):
            df = parse_excel(content)
        elif ext == "sav":
            df = parse_spss(content)
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /upload/preview: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")
    return {
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


@app.post("/run", response_model=ModelResult)
async def run_model(
    file: UploadFile = File(...),
    model: str = Form(...),
    algorithm: str = Form("pls"),
    bootstrap_n: int = Form(0),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)

    raw = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    log("step", f"Parsing uploaded file: {file.filename}")
    try:
        if ext == "csv":
            df = parse_csv_robust(raw)
        elif ext in ("xlsx", "xls"):
            df = parse_excel(raw)
        elif ext == "sav":
            df = parse_spss(raw)
        else:
            if run_id in _run_store: _run_store[run_id]["done"] = True
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as e:
        logger.error("File parse error in /run: %s", e, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        log("info", "Missing data: mean imputation applied")

    df = auto_reverse_score(df, model, log_fn=log)
    model = _expand_covariances(model)   # TC-52: expand 'y2 ~~ y4 + y6' → individual pairs
    # Run blocking computation in thread executor so SSE stream stays live
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: fit_model(df, model, algorithm=algorithm,
                               bootstrap_n=bootstrap_n, log_fn=log)
        )
    except ValueError as e:
        log("error", f"Model fit failed: {e}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(e))
    except Exception as e:
        log("error", "Unexpected engine error — see server logs for details")
        logger.error("Unexpected engine error in /run: %s", e, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "Model fitting failed. Check server logs.")

    # Fingerprint
    try:
        fp, audit = _compute_fingerprint(run_id, model, df, algorithm, result)
        result.run_id = run_id
        result.fingerprint = fp
        if run_id in _run_store: _run_store[run_id]["fingerprint"] = fp
        if run_id in _run_store: _run_store[run_id]["audit"] = audit
        log("ok", f"Fingerprint: {fp[:16]}…{fp[-8:]}")
    except Exception as e:
        log("warn", f"Fingerprint computation failed: {e}")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


@app.post("/bootstrap", response_model=BootstrapResult)
async def bootstrap_only(
    file: UploadFile = File(...),
    model: str = Form(...),
    bootstrap_n: int = Form(500),
    algorithm: str = Form("pls"),
):
    bootstrap_n = min(bootstrap_n, 20_000)
    content = await file.read()
    try:
        df = _parse_upload(content, file.filename)
        df = df.dropna()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /bootstrap: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")
    df = auto_reverse_score(df, model)   # TC-67: parity with /run — must reverse-score before fitting
    model = _expand_covariances(model)   # TC-52: expand multi-target ~~ before engine sees the model
    try:
        return run_bootstrap(df, model, n=bootstrap_n, algorithm=algorithm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /bootstrap: %s", e, exc_info=True)
        raise HTTPException(500, "Bootstrap analysis failed. Check server logs.")


@app.post("/htmt", response_model=HTMTResult)
async def htmt(file: UploadFile = File(...), model: str = Form(...)):
    content = await file.read()
    try:
        df = _parse_upload(content, file.filename)
        df = df.dropna()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /htmt: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")
    df = auto_reverse_score(df, model)
    try:
        return compute_htmt(df, model)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /htmt: %s", e, exc_info=True)
        raise HTTPException(500, "HTMT computation failed. Check server logs.")


@app.post("/validate-syntax")
async def validate_syntax(payload: dict):
    model = payload.get("model", "")
    try:
        parsed = parse_lavaan(model)
        # TC-64: a non-empty string that contains no SEM operators (=~, ~, ~~)
        # parses without raising but produces empty dicts — treat as invalid.
        if not parsed.get("measurement") and not parsed.get("structural"):
            return {
                "valid": False,
                "error": "No SEM operators (=~, ~) found in model syntax.",
            }
        return {"valid": True, "parsed": parsed}
    except ValueError as e:
        # Expected: user submitted syntactically incomplete/invalid lavaan syntax.
        # Log at WARNING — this is routine input validation, not a server fault.
        logger.warning("Model syntax validation failed: %s", e)
        return {
            "valid": False,
            "error": "Invalid model syntax: " + e.args[0] if e.args else "Invalid model syntax.",          
        }
    except Exception as e:
        # Unexpected: a real server-side failure (programming error, OOM, etc.)
        logger.exception("Unexpected error in validate_syntax")
        return {
            "valid": False,
            "error": "Model validation failed.",
        }


@app.post("/export")
async def export_code(payload: dict):
    """
    Export the model as runnable code.

    Body:
      {
        "model":     "<lavaan syntax>",
        "algorithm": "pls" | "cb" | "wls",
        "format":    "r" | "python" | "lav"
      }

    Returns a plain-text file download.
    """
    model = payload.get("model", "")
    algorithm = payload.get("algorithm", "pls")
    fmt = payload.get("format", "r")

    if not model.strip():
        raise HTTPException(400, "No model syntax provided.")

    try:
        code = export_as_code(model, algorithm=algorithm, format=fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("Unexpected error in /export: %s", e, exc_info=True)
        raise HTTPException(500, "Code export failed. Check server logs.")

    ext_map = {"r": "R", "python": "py", "lav": "lav"}
    ext = ext_map.get(fmt, "txt")

    return PlainTextResponse(
        content=code,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="naval_sem_model.{ext}"'},
    )


# ── v0.6: Multi-Group Analysis ─────────────────────────────────────────────────

def _parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    """Shared helper: parse uploaded file bytes to DataFrame."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return parse_csv_robust(content)
    elif ext in ("xlsx", "xls"):
        return parse_excel(content)
    elif ext == "sav":
        return parse_spss(content)
    raise HTTPException(400, f"Unsupported file type: .{ext}")


@app.post("/mga", response_model=MGAResult)
async def multi_group_analysis(
    file: UploadFile = File(...),
    model: str = Form(...),
    group_col: str = Form(...),
    algorithm: str = Form("pls"),
    bootstrap_n: int = Form(500),
    n_permutations: int = Form(500),
    run_micom: bool = Form(True),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    """
    Multi-Group Analysis (MGA) with optional MICOM measurement invariance test.

    Form fields
    -----------
    file          : CSV, XLSX, or SAV dataset.
    model         : lavaan-style model syntax (same model applied to all groups).
    group_col     : Column used to split the dataset into groups.
                    Values are stringified.  Max 10 distinct groups.
    algorithm     : ``pls`` (default) | ``cb`` | ``wls``.
    bootstrap_n   : Bootstrap resamples for per-pair path-difference CIs (default 500).
    n_permutations: Permutation samples for MICOM steps 2 and 3 (default 500).
    run_micom     : Whether to run MICOM before MGA (2-group PLS only, default True).
    missing       : ``listwise`` (default) | ``mean``.
    run_id        : Optional SSE tracking ID — logs available at /logs/{run_id}.

    Returns
    -------
    MGAResult — per-group fit, pairwise path-difference CIs, optional MICOM.
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n    = min(bootstrap_n, 20_000)
    n_permutations = min(n_permutations, 20_000)

    raw = await file.read()
    log("step", f"MGA: parsing uploaded file: {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /mga: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if group_col not in df.columns:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(
            422,
            f"Group column '{group_col}' not found. "
            f"Available columns: {df.columns.tolist()}",
        )

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        log("info", "Missing data: mean imputation applied")

    df = auto_reverse_score(df, model, log_fn=log)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_mga(
                df, model,
                group_col=group_col,
                algorithm=algorithm,
                bootstrap_n=bootstrap_n,
                n_permutations=n_permutations,
                run_micom_test=run_micom,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"MGA failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "MGA unexpected error — see server logs for details")
        logger.error("Unexpected error in /mga: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "MGA analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


# ── v0.6: Higher-Order Constructs ──────────────────────────────────────────────

@app.post("/hoc", response_model=ModelResult)
async def hoc_analysis(
    file: UploadFile = File(...),
    model: str = Form(...),
    hoc_method: str = Form("repeated_indicator"),
    algorithm: str = Form("pls"),
    bootstrap_n: int = Form(0),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    """
    Higher-Order Construct (HOC) estimation.

    Automatically detects HOCs from the lavaan syntax: any latent variable
    whose measurement block contains the names of other latent variables is
    treated as a HOC.

    Form fields
    -----------
    file          : CSV, XLSX, or SAV dataset.
    model         : lavaan-style model syntax including HOC definitions.
                    Example::

                        HOC  =~ FOC1 + FOC2
                        FOC1 =~ x1 + x2 + x3
                        FOC2 =~ x4 + x5 + x6
                        Y    ~  HOC

    hoc_method    : ``repeated_indicator`` (default) | ``two_stage``.
    algorithm     : ``pls`` (default) | ``cb`` | ``wls``.
                    Note: ``two_stage`` always uses PLS for Stage 1 regardless
                    of this setting.
    bootstrap_n   : Bootstrap resamples (default 0 = no bootstrap).
    missing       : ``listwise`` (default) | ``mean``.
    run_id        : Optional SSE tracking ID.

    Returns
    -------
    ModelResult with ``hoc_type`` set to the method used.
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)

    raw = await file.read()
    log("step", f"HOC: parsing uploaded file: {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /hoc: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        log("info", "Missing data: mean imputation applied")

    df = auto_reverse_score(df, model, log_fn=log)

    # TC-68: reject models that contain no higher-order constructs before hitting the engine.
    # A HOC is any latent variable whose measurement block lists other latent variables as indicators.
    try:
        _parsed = parse_lavaan(model)
        _latent = set(_parsed.get("latent_vars", []))
        _measurement = _parsed.get("measurement", {})
        _has_hoc = any(
            any(ind in _latent for ind in indicators)
            for indicators in _measurement.values()
        )
    except Exception:
        _has_hoc = True  # if we can't parse, let the engine decide
    if not _has_hoc:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(
            422,
            "No higher-order constructs detected in model syntax. "
            "Use /run for standard (first-order) models.",
        )

    if hoc_method not in ("repeated_indicator", "two_stage"):
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(
            400,
            f"Unknown hoc_method '{hoc_method}'. "
            "Use 'repeated_indicator' or 'two_stage'.",
        )

    loop = asyncio.get_event_loop()
    try:
        fn = (
            fit_hoc_repeated_indicator
            if hoc_method == "repeated_indicator"
            else fit_hoc_two_stage
        )
        result = await loop.run_in_executor(
            None,
            lambda: fn(
                df, model,
                algorithm=algorithm,
                bootstrap_n=bootstrap_n,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"HOC failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "HOC unexpected error — see server logs for details")
        logger.error("Unexpected error in /hoc: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "HOC analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


# ── v0.7: Moderation ──────────────────────────────────────────────────────────

@app.post("/moderation", response_model=ModerationResult)
async def moderation_analysis(
    file:        UploadFile = File(...),
    model:       str        = Form(...),
    algorithm:   str        = Form("pls"),
    bootstrap_n: int        = Form(500),
    missing:     str        = Form("listwise"),
    run_id:      str        = Form(None),
):
    """
    Moderation analysis via the product-of-composites approach.

    Detects ``X*M`` interaction terms in the lavaan structural syntax.

    Form fields
    -----------
    file        : CSV, XLSX, or SAV dataset.
    model       : lavaan syntax with at least one ``X*M`` interaction term.
                  Example::

                      Y  ~  X + M + X*M
                      X  =~ x1 + x2 + x3
                      M  =~ m1 + m2 + m3
                      Y  =~ y1 + y2 + y3

    algorithm   : ``pls`` (default) | ``cb`` | ``wls``.
    bootstrap_n : Bootstrap resamples for simple-slope CIs (default 500).
    missing     : ``listwise`` (default) | ``mean``.
    run_id      : Optional SSE tracking ID.

    Returns
    -------
    ModerationResult — interaction β with CI, Δ R², f², simple slopes at
    −1 SD / mean / +1 SD of the moderator.
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)

    raw = await file.read()
    log("step", f"Moderation: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /moderation: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model, log_fn=log)

    # TC-40: manifest-variable models have no =~ lines — run_moderation() calls the
    # PLS engine which requires at least one latent variable → guaranteed crash.
    # Detect this up front and route to the OLS fallback instead.
    try:
        _mod_parsed = parse_lavaan(model)
        _is_manifest_only = not _mod_parsed.get("measurement")
    except Exception:
        _is_manifest_only = False

    if _is_manifest_only:
        log("info", "Manifest-variable moderation model (no =~) — using OLS fallback.")
        try:
            _manifest_result = _manifest_moderation(df, model, bootstrap_n=bootstrap_n, log_fn=log)
        except Exception as exc:
            logger.error("OLS manifest moderation failed: %s", exc, exc_info=True)
            if run_id in _run_store: _run_store[run_id]["done"] = True
            raise HTTPException(422, f"Manifest moderation failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        return JSONResponse(content=_manifest_result)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_moderation(
                df, model,
                algorithm=algorithm,
                bootstrap_n=bootstrap_n,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"Moderation failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "Moderation unexpected error — see server logs for details")
        logger.error("Unexpected error in /moderation: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "Moderation analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


# ── v0.7: IPMA ────────────────────────────────────────────────────────────────

@app.post("/ipma", response_model=IPMAResult)
async def ipma_analysis(
    file:      UploadFile    = File(...),
    model:     str           = Form(...),
    target_lv: str           = Form(...),
    algorithm: str           = Form("pls"),
    scale_min: float         = Form(None),
    scale_max: float         = Form(None),
    missing:   str           = Form("listwise"),
    run_id:    str           = Form(None),
):
    """
    Importance-Performance Map Analysis (IPMA).

    Ringle & Sarstedt (2016) / Hair et al. (2022).

    Form fields
    -----------
    file       : CSV, XLSX, or SAV dataset.
    model      : lavaan syntax.
    target_lv  : The dependent LV for which importance is computed
                 (e.g. ``"Loyalty"``).
    algorithm  : ``pls`` (default) | ``cb`` | ``wls``.
    scale_min  : Theoretical scale minimum (e.g. ``1`` for Likert 1-5).
                 If omitted, the observed minimum composite score is used.
    scale_max  : Theoretical scale maximum (e.g. ``5`` for Likert 1-5).
    missing    : ``listwise`` (default) | ``mean``.
    run_id     : Optional SSE tracking ID.

    Returns
    -------
    IPMAResult — importance (total effect) and performance (0–100 rescaled
    composite mean) for each predictor of ``target_lv``, sorted by importance.
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)

    raw = await file.read()
    log("step", f"IPMA: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /ipma: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model, log_fn=log)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: compute_ipma(
                df, model,
                target_lv=target_lv,
                algorithm=algorithm,
                scale_min=scale_min,
                scale_max=scale_max,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"IPMA failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "IPMA unexpected error — see server logs for details")
        logger.error("Unexpected error in /ipma: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "IPMA analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


# ── v0.7: NCA ─────────────────────────────────────────────────────────────────

@app.post("/nca", response_model=NCAResult)
async def nca_analysis(
    file:            UploadFile = File(...),
    model:           str        = Form(...),
    n_permutations:  int        = Form(1000),
    missing:         str        = Form("listwise"),
    run_id:          str        = Form(None),
):
    """
    Necessary Condition Analysis (NCA).

    Dul (2016, 2020) — CE-FDH and CR-FDH ceiling lines and effect size d.

    Form fields
    -----------
    file           : CSV, XLSX, or SAV dataset.
    model          : lavaan syntax. All structural IV → DV pairs are tested.
    n_permutations : Permutation samples for significance test (default 1000).
    missing        : ``listwise`` (default) | ``mean``.
    run_id         : Optional SSE tracking ID.

    Returns
    -------
    NCAResult — CE-FDH d, CR-FDH d, ceiling line coordinates (for scatter
    plot rendering), and permutation p-values for each IV → DV pair.

    Effect size benchmarks (Dul 2016)
    ----------------------------------
    d < 0.1   negligible
    d < 0.3   small
    d < 0.5   medium
    d ≥ 0.5   large
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    n_permutations = min(n_permutations, 20_000)

    raw = await file.read()
    log("step", f"NCA: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /nca: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model, log_fn=log)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: compute_nca(
                df, model,
                n_permutations=n_permutations,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"NCA failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "NCA unexpected error — see server logs for details")
        logger.error("Unexpected error in /nca: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "NCA analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result


# ── v0.7: Moderated Mediation ─────────────────────────────────────────────────

@app.post("/mod-mediation", response_model=ModMediationResult)
async def mod_mediation_analysis(
    file:        UploadFile = File(...),
    model:       str        = Form(...),
    algorithm:   str        = Form("pls"),
    bootstrap_n: int        = Form(500),
    missing:     str        = Form("listwise"),
    run_id:      str        = Form(None),
):
    """
    Moderated Mediation / Conditional Process Analysis.

    Edwards & Lambert (2007); Hayes (2018, Chapters 11–14).

    Detect interaction terms (``X*W``) in the structural syntax and compute:
      - Path coefficients a, b, c', and the interaction β.
      - Index of Moderated Mediation (IMM) with bootstrap 95 % CI.
      - Conditional indirect effects at W = −1 SD, mean (0), +1 SD.

    Supported Hayes PROCESS patterns
    ---------------------------------
    a-path moderation (Process Model 7):
        M ~ X + W + X*W
        Y ~ X + M

    b-path moderation (Process Model 14):
        M ~ X
        Y ~ X + M + W + M*W

    Both paths (Process Model 58/59): combine both interaction terms.

    Form fields
    -----------
    file        : CSV, XLSX, or SAV dataset.
    model       : lavaan syntax with at least one ``X*W`` interaction term.
    algorithm   : ``pls`` (default) | ``cb`` | ``wls``.
    bootstrap_n : Bootstrap resamples for IMM and conditional IE CIs (default 500).
    missing     : ``listwise`` (default) | ``mean``.
    run_id      : Optional SSE tracking ID.

    Returns
    -------
    ModMediationResult — one ModMediationPath entry per detected X→M→Y chain.
    """
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)

    raw = await file.read()
    log("step", f"ModMediation: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        logger.error("File parse error in /mod-mediation: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    df = auto_reverse_score(df, model, log_fn=log)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_mod_mediation(
                df, model,
                algorithm=algorithm,
                bootstrap_n=bootstrap_n,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"ModMediation failed: {exc}")
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", "ModMediation unexpected error — see server logs for details")
        logger.error("Unexpected error in /mod-mediation: %s", exc, exc_info=True)
        if run_id in _run_store: _run_store[run_id]["done"] = True
        raise HTTPException(500, "Moderated mediation analysis failed. Check server logs.")

    if run_id in _run_store: _run_store[run_id]["done"] = True
    return result
