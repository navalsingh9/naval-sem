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
        "naval_sem_version": "0.6.0",
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
    CMBMarkerResult, PredictResult,
)

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

app = FastAPI(title="NAVAL-SEM API", version="0.5.0")

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
    return {"status": "ok", "version": "0.5.0"}


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
