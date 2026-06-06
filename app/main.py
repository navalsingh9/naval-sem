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
    # Prune old runs (keep last 50) to avoid unbounded memory
    if len(_run_store) > 50:
        oldest = sorted(_run_store, key=lambda k: _run_store[k]["logs"][0]["t"]
                        if _run_store[k]["logs"] else 0)[:len(_run_store) - 50]
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
)
from app.engine_mga import run_mga, fit_hoc_repeated_indicator, fit_hoc_two_stage
from app.engine_v07 import run_moderation, compute_ipma                           # v0.7
from app.nca       import compute_nca                                              # v0.7

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
        content={"detail": f"Backend generated an invalid response (likely NaN estimates). Details: {str(exc)[:300]}"}
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
        return JSONResponse({"error": f"Static folder not found: {_STATIC_DIR}"}, status_code=500)



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
        while elapsed < 3600:
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
        yield f"data: {json.dumps({'level': 'error', 'msg': 'SSE timeout (60 min)'})}\n\n"

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
        with urlopen(req, timeout=8) as resp:
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
        logger.warning(f"check-updates failed: {exc}")
        return {"current_version": APP_VERSION, "status": "error", "detail": str(exc)}


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
        raise HTTPException(422, f"File parse error: {e}")

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
        raise HTTPException(500, f"Predictive relevance error: {e}")


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
        raise HTTPException(422, f"File parse error: {e}")

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
        raise HTTPException(500, f"CMB analysis error: {e}")


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
        raise HTTPException(422, f"File parse error: {e}")

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
        raise HTTPException(500, f"Indirect effects error: {e}")


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
        raise HTTPException(422, f"Could not parse file: {e}")
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
            _run_store[run_id]["done"] = True
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        _run_store[run_id]["done"] = True
        raise
    except Exception as e:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {e}")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        log("info", "Missing data: mean imputation applied")

    df = auto_reverse_score(df, model, log_fn=log)
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
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(e))
    except Exception as e:
        log("error", f"Unexpected engine error: {e}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"Model fitting error: {e}")

    # Fingerprint
    try:
        fp, audit = _compute_fingerprint(run_id, model, df, algorithm, result)
        result.run_id = run_id
        result.fingerprint = fp
        _run_store[run_id]["fingerprint"] = fp
        _run_store[run_id]["audit"] = audit
        log("ok", f"Fingerprint: {fp[:16]}…{fp[-8:]}")
    except Exception as e:
        log("warn", f"Fingerprint computation failed: {e}")

    _run_store[run_id]["done"] = True
    return result


@app.post("/bootstrap", response_model=BootstrapResult)
async def bootstrap_only(
    file: UploadFile = File(...),
    model: str = Form(...),
    n: int = Form(500),
    algorithm: str = Form("pls"),
):
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        df = parse_csv_robust(content) if ext == "csv" else parse_excel(content)
        df = df.dropna()
    except Exception as e:
        raise HTTPException(422, str(e))
    try:
        return run_bootstrap(df, model, n=n, algorithm=algorithm)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/htmt", response_model=HTMTResult)
async def htmt(file: UploadFile = File(...), model: str = Form(...)):
    content = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()
    try:
        df = parse_csv_robust(content) if ext == "csv" else parse_excel(content)
        df = df.dropna()
    except Exception as e:
        raise HTTPException(422, str(e))
    df = auto_reverse_score(df, model)
    try:
        return compute_htmt(df, model)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/validate-syntax")
async def validate_syntax(payload: dict):
    model = payload.get("model", "")
    try:
        parsed = parse_lavaan(model)
        return {"valid": True, "parsed": parsed}
    except Exception as e:
        logger.exception("Model validation error")
        return {
            "valid": False,
            "error": "Model validation failed."
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
        raise HTTPException(500, f"Export error: {e}")

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

    raw = await file.read()
    log("step", f"MGA: parsing uploaded file: {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {exc}")

    if group_col not in df.columns:
        _run_store[run_id]["done"] = True
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
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", f"MGA unexpected error: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"MGA error: {exc}")

    _run_store[run_id]["done"] = True
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

    raw = await file.read()
    log("step", f"HOC: parsing uploaded file: {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {exc}")

    if missing == "listwise":
        df = df.dropna()
        log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        log("info", "Missing data: mean imputation applied")

    df = auto_reverse_score(df, model, log_fn=log)

    if hoc_method not in ("repeated_indicator", "two_stage"):
        _run_store[run_id]["done"] = True
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
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", f"HOC unexpected error: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"HOC error: {exc}")

    _run_store[run_id]["done"] = True
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

    raw = await file.read()
    log("step", f"Moderation: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {exc}")

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
            lambda: run_moderation(
                df, model,
                algorithm=algorithm,
                bootstrap_n=bootstrap_n,
                log_fn=log,
            ),
        )
    except ValueError as exc:
        log("error", f"Moderation failed: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", f"Moderation unexpected error: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"Moderation error: {exc}")

    _run_store[run_id]["done"] = True
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
        _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {exc}")

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
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", f"IPMA unexpected error: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"IPMA error: {exc}")

    _run_store[run_id]["done"] = True
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

    raw = await file.read()
    log("step", f"NCA: parsing {file.filename}")
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        _run_store[run_id]["done"] = True
        raise
    except Exception as exc:
        _run_store[run_id]["done"] = True
        raise HTTPException(422, f"File parse error: {exc}")

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
        _run_store[run_id]["done"] = True
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log("error", f"NCA unexpected error: {exc}")
        _run_store[run_id]["done"] = True
        raise HTTPException(500, f"NCA error: {exc}")

    _run_store[run_id]["done"] = True
    return result
