"""
NAVAL-SEM — FastAPI Backend + Static Frontend Server
Run standalone: uvicorn app.main:app --reload --port 8000
Run via launcher: python launcher.py
"""

import os
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from app.engine import fit_model, run_bootstrap, compute_htmt, export_as_code, compute_indirect_effects
from app.parser import (parse_spss, parse_excel, parse_lavaan,parse_csv_robust,)
from app.schemas import ModelResult, BootstrapResult, HTMTResult, IndirectResult

_STATIC_DIR = os.environ.get(
    "NAVAL_SEM_STATIC",
    str(Path(__file__).parent.parent / "static"),
)

app = FastAPI(title="NAVAL-SEM API", version="0.4.0")

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


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4.0"}


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
):
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
    try:
        result = fit_model(df, model, algorithm=algorithm, bootstrap_n=bootstrap_n)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Model fitting error: {e}")
    if bootstrap_n > 0:
        try:
            result.bootstrap = run_bootstrap(df, model, n=bootstrap_n, algorithm=algorithm)
        except Exception as e:
            result.bootstrap_error = str(e)
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
