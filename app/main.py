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
import datetime
import threading
from pathlib import Path
import logging
from contextlib import contextmanager
from typing import Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

# ── Single source of truth for the app version ───────────────────────────────
from app.version import APP_VERSION

_GITHUB_REPO = "navalsingh9/naval-sem"


# ── Per-run log store ─────────────────────────────────────────────────────────
# Keyed by run_id; entries are {"t": ms_epoch, "level": str, "msg": str}
# "done" is set True once the run completes so the SSE stream can terminate.
_run_store: dict[str, dict] = {}
_run_store_lock = threading.RLock()

import re as _re
_RUN_ID_RE = _re.compile(r'^[0-9a-f\-]{8,36}$')

def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id format")
    return run_id


def _init_run(run_id: str):
    with _run_store_lock:
        _run_store[run_id] = {"logs": [], "done": False, "fingerprint": None, "audit": None}
        if len(_run_store) > 500:
            all_keys = sorted(
                (k for k in _run_store if k != run_id),
                key=lambda k: _run_store[k]["logs"][0]["t"] if _run_store[k]["logs"] else 0
            )
            for k in all_keys[:300]:
                del _run_store[k]
        elif len(_run_store) > 200:
            completed = [k for k in _run_store if k != run_id and _run_store[k]["done"]]
            oldest = sorted(completed, key=lambda k: _run_store[k]["logs"][0]["t"] if _run_store[k]["logs"] else 0)
            for k in oldest[:max(0, len(_run_store) - 200)]:
                del _run_store[k]


def _make_log_fn(run_id: str):
    """Return a (level, msg) → None callback that appends to _run_store."""
    def log_fn(level: str, msg: str):
        with _run_store_lock:
            if run_id in _run_store:
                _run_store[run_id]["logs"].append({
                    "t": round(time.time() * 1000),
                    "level": level.lower(),
                    "msg": msg,
                })
    return log_fn


@contextmanager
def _run_context(run_id: str):
    """Guarantee _run_store[run_id]['done'] = True on exit regardless of how the route exits."""
    try:
        yield
    finally:
        with _run_store_lock:
            if run_id in _run_store:
                _run_store[run_id]["done"] = True


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
    compute_nomological_validity, compute_measurement_invariance,
)
from app.scale import compute_efa, compute_cvi
from app.parser import (parse_spss, parse_excel, parse_lavaan, parse_csv_robust,)
from app.schemas import (
    ModelResult, BootstrapResult, HTMTResult, IndirectResult,
    CMBMarkerResult, PredictResult, MGAResult,
    ModerationResult, IPMAResult, NCAResult,          # v0.7
    NCAESSEResult,                                    # v0.9
    ModMediationResult,                               # v0.7 — moderated mediation
    RobustnessChecks, FIMIXResult, PLSPOSResult,      # v0.8
    GaussianCopulaResult, NonlinearResult,            # v0.8
    NomologicalResult,                                # v0.9
    MeasurementInvarianceResult,                      # v0.9
    CVIResult, ScaleDevelopmentResult,                # v0.9 — scale development
    FsQCAResult,                                       # v1.0 — fsQCA
)
from app.engine_mga import run_mga, fit_hoc_repeated_indicator, fit_hoc_two_stage
from app.engine_moderation    import run_moderation                                # v0.7
from app.engine_ipma          import compute_ipma                                  # v0.7
from app.nca                  import compute_nca                                   # v0.7
from app.nca_esse             import compute_nca_esse                              # v0.9
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
                    "ci_lower": None, "ci_upper": None, "significant": False,
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
                    "significant": (
                        bool(ci_lower_95 is not None and ci_upper_95 is not None
                             and (ci_lower_95 > 0 or ci_upper_95 < 0))
                        if bootstrap_n > 0 else None
                    ),
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
                    "ci_lower": None, "ci_upper": None, "significant": False,
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


def auto_reverse_score(df: pd.DataFrame, model_syntax: str, log_fn=None, reverse_items: Optional[str] = "") -> pd.DataFrame:
    """
    Reverse-scores variables before model fitting.

    Explicit mode (preferred): pass ``reverse_items='rVAR1,rVAR2'``.
      Only the listed names are reverse-scored.  Each name must start with
      'r' and the base variable (name[1:]) must exist in the DataFrame.

    Legacy heuristic mode (no reverse_items supplied):
      Scans the model syntax for 'r'-prefixed observed variables and
      reverse-scores them automatically.  Emits a deprecation warning
      encouraging callers to switch to the explicit form.
    """
    df = df.copy()  # prevent in-place mutation of the caller's DataFrame

    if reverse_items is None:
        reverse_items = ""

    explicit = [v.strip() for v in reverse_items.split(",") if v.strip()]

    if explicit:
        # ── Explicit mode ────────────────────────────────────────────────────
        for name in explicit:
            if name not in df.columns and len(name) > 1 and name[1:] in df.columns:
                base = name[1:]
                col_max, col_min = df[base].max(), df[base].min()
                df[name] = (col_max + col_min) - df[base]
                if log_fn:
                    log_fn("info", f"Reverse scored '{base}' into '{name}'")
    else:
        # ── Legacy r-prefix heuristic ────────────────────────────────────────
        if log_fn:
            log_fn("warn",
                   "auto_reverse_score: using implicit r-prefix heuristic. "
                   "Pass reverse_items='rVAR1,rVAR2' to be explicit and suppress this warning.")
        try:
            parsed = parse_lavaan(model_syntax)
            observed_vars = parsed.get("observed_vars", [])
        except Exception:
            return df  # If syntax is broken, do nothing and let the engine catch it later

        for var in observed_vars:
            if var not in df.columns:
                if var.startswith('r') and var[1:] in df.columns:
                    base_var = var[1:]
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

@app.on_event("startup")
async def _startup_warnings():
    import socket
    host = os.getenv("HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "CORS is open (allow_origins=['*']) and this server is bound to %s. "
            "The API has no authentication — do not expose to untrusted networks.",
            host
        )
    workers = int(os.environ.get("WEB_CONCURRENCY", os.environ.get("NAVAL_SEM_WORKERS", "1")))
    if workers > 1:
        logger.warning(
            "NAVAL-SEM: _run_store is process-local. SSE streaming (/logs/{run_id}) "
            "requires single-worker deployment (WEB_CONCURRENCY=1). "
            "For multi-worker setups, replace _run_store with a shared store (Redis)."
        )

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
# CORS: allow_origins=["*"] is intentional for single-user local deployments.
# If deploying multi-tenant or over a network, restrict origins and add auth.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Static files mounted at end of file — see bottom of module.



@app.get("/logs/{run_id}")
async def stream_logs(run_id: str):
    """
    Server-Sent Events stream for a running analysis.
    The client opens this immediately after submitting /run, passing the same run_id.
    Streams {"t","level","msg"} entries; terminates with {"done":true,"fingerprint":"..."}.
    """
    _validate_run_id(run_id)
    async def event_gen():
        wait_time = 0.0
        while True:
            with _run_store_lock:
                _exists = _run_store.get(run_id) is not None
            if _exists or wait_time >= 5.0:
                break
            await asyncio.sleep(0.5)
            wait_time += 0.5
        last = 0
        elapsed = 0.0
        while elapsed < 14400:
            with _run_store_lock:
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
    _validate_run_id(run_id)
    with _run_store_lock:
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
    reverse_items: Optional[str] = Form(None),
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

    df = auto_reverse_score(df, model, reverse_items=reverse_items)
    model = _expand_covariances(model)   # M10: expand multi-target ~~ before engine sees the model
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
    reverse_items: Optional[str] = Form(None),
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

    df = auto_reverse_score(df, model, reverse_items=reverse_items)
    model = _expand_covariances(model)
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
    algorithm: str = Form("pls"),
):
    """
    Decompose indirect (mediation) effects for all variable pairs connected
    via paths of length ≥ 2. Returns indirect effects with bootstrapped 95% CIs
    and a total effects matrix (direct + indirect).

    Supports ``algorithm=cb`` for observed-variable (Hayes PROCESS-style) path
    models — no ``=~`` measurement blocks required when using CB-SEM.
    """
    bootstrap_n = min(bootstrap_n, 20_000)
    if algorithm not in ("pls", "cb", "wls"):
        raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")
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
        return compute_indirect_effects(df, model, n_bootstrap=bootstrap_n, algorithm=algorithm)
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
    reverse_items: Optional[str] = Form(None),
):
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

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
                raise HTTPException(400, f"Unsupported file type: {ext}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("File parse error in /run: %s", e, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))
            log("info", "Missing data: mean imputation applied")

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)
        model = _expand_covariances(model)   # TC-52: expand 'y2 ~~ y4 + y6' → individual pairs
        # Run blocking computation in thread executor so SSE stream stays live
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: fit_model(df, model, algorithm=algorithm,
                                   bootstrap_n=bootstrap_n, log_fn=log)
            )
        except ValueError as e:
            log("error", f"Model fit failed: {e}")
            raise HTTPException(422, str(e))
        except Exception as e:
            log("error", "Unexpected engine error — see server logs for details")
            logger.error("Unexpected engine error in /run: %s", e, exc_info=True)
            raise HTTPException(500, "Model fitting failed. Check server logs.")

        # Fingerprint
        try:
            fp, audit = _compute_fingerprint(run_id, model, df, algorithm, result)
            result.run_id = run_id
            result.fingerprint = fp
            with _run_store_lock:
                if run_id in _run_store:
                    _run_store[run_id]["fingerprint"] = fp
                    _run_store[run_id]["audit"] = audit
            log("ok", f"Fingerprint: {fp[:16]}…{fp[-8:]}")
        except Exception as e:
            log("warn", f"Fingerprint computation failed: {e}")

        return result


@app.post("/bootstrap", response_model=BootstrapResult)
async def bootstrap_only(
    file: UploadFile = File(...),
    model: str = Form(...),
    bootstrap_n: int = Form(500),
    algorithm: str = Form("pls"),
    reverse_items: Optional[str] = Form(None),
):
    bootstrap_n = min(bootstrap_n, 20_000)
    if algorithm not in ("pls", "cb", "wls"):
        raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")
    content = await file.read()
    try:
        df = _parse_upload(content, file.filename)
        df = df.dropna()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /bootstrap: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")
    df = auto_reverse_score(df, model, reverse_items=reverse_items)   # TC-67: parity with /run — must reverse-score before fitting
    model = _expand_covariances(model)   # TC-52: expand multi-target ~~ before engine sees the model
    try:
        return run_bootstrap(df, model, n=bootstrap_n, algorithm=algorithm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Unexpected error in /bootstrap: %s", e, exc_info=True)
        raise HTTPException(500, "Bootstrap analysis failed. Check server logs.")


@app.post("/htmt", response_model=HTMTResult)
async def htmt(file: UploadFile = File(...), model: str = Form(...), reverse_items: Optional[str] = Form(None)):
    content = await file.read()
    try:
        df = _parse_upload(content, file.filename)
        df = df.dropna()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File parse error in /htmt: %s", e, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")
    df = auto_reverse_score(df, model, reverse_items=reverse_items)
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


@app.post("/export/pdf")
async def export_pdf(payload: dict):
    """
    Generate a ReportLab PDF report from the frontend analysis snapshot.

    Body (all keys optional except snap):
      snap         — frozen snapshot  {runId, ts, algo, bsN, miss, fname, cmb,
                                       syntax, analysisType, n_obs, n_params}
      results      — full ModelResult dict
      mga          — MGAResult dict
      htmt         — HTMTResult dict
      predictive   — PredictResult dict
      diagram_png  — base64-encoded PNG of the path diagram
      analyst      — {name, email, org}
      note         — analyst note string

    Returns a binary PDF file download.
    Requires:  pip install reportlab

    Top-level try/except: every code path below this point used to have its
    own defensive try/except, but the response-construction tail (run_id
    validation, filename formatting, StreamingResponse) had none — any
    exception there escaped FastAPI entirely and surfaced to the client as
    Starlette's generic, detail-free "Internal Server Error" plain-text
    page (no JSON, no traceback, undiagnosable from the client side). This
    outer guard ensures *every* failure mode returns an HTTPException with
    real diagnostic detail instead.
    """
    try:
        try:
            from app.export_pdf import generate_pdf
        except Exception:
            # Catches ImportError *and* anything export_pdf.py's module-level
            # code might raise on import (font/constant setup, etc.) — a bare
            # `except ImportError` here means any other exception type escapes
            # this function entirely and becomes FastAPI's generic, detail-free
            # 500 page, which is undiagnosable from the client side.
            try:
                import importlib.util, pathlib
                spec = importlib.util.spec_from_file_location(
                    "export_pdf",
                    pathlib.Path(__file__).parent / "export_pdf.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                generate_pdf = mod.generate_pdf
            except Exception as ie:
                logger.error("export_pdf import failed: %s", ie, exc_info=True)
                raise HTTPException(
                    500,
                    f"export_pdf module failed to load: {ie!r}. "
                    "Check server logs for the full traceback; if reportlab "
                    "is missing, install it with: pip install reportlab"
                ) from ie

        try:
            pdf_bytes = await asyncio.get_running_loop().run_in_executor(
                None, lambda: generate_pdf(payload)
            )
        except Exception as e:
            # exc_info=True can itself raise on some Windows console code pages
            # if the formatted message contains non-ASCII characters — never let
            # a logging failure mask the real error with an unhandled exception.
            try:
                logger.error("PDF generation error: %r", e, exc_info=True)
            except Exception:
                logger.error("PDF generation error (unprintable): %s", type(e).__name__)
            # Build the HTTPException detail defensively too — str(e) can
            # itself contain non-ASCII content that the same Windows console
            # encoding issue would choke on when uvicorn's access logger
            # later tries to print the response; ASCII-escape it so the
            # error is always at least visible, even if not pretty.
            try:
                detail = f"PDF generation failed: {e!r}"
            except Exception:
                detail = f"PDF generation failed: {type(e).__name__} (unprintable message)"
            raise HTTPException(500, detail.encode("ascii", "backslashreplace").decode("ascii"))

        snap     = payload.get("snap") or {}
        run_id   = (snap.get("runId") or "report")[:36]
        try:
            _validate_run_id(run_id)
        except HTTPException:
            run_id = "report"
        safe_fname = _re.sub(r'[^a-zA-Z0-9\-]', '', run_id)[:8]
        ts_stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"naval_sem_report_{safe_fname}_{ts_stamp}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Final safety net: anything that reaches here would otherwise have
        # escaped as Starlette's generic "Internal Server Error" plain-text
        # page with zero diagnostic value. Always return real detail instead.
        try:
            logger.error("Unhandled /export/pdf failure: %r", exc, exc_info=True)
        except Exception:
            logger.error("Unhandled /export/pdf failure (unprintable): %s", type(exc).__name__)
        try:
            detail = f"/export/pdf failed unexpectedly: {type(exc).__name__}: {exc!r}"
        except Exception:
            detail = f"/export/pdf failed unexpectedly: {type(exc).__name__} (unprintable message)"
        raise HTTPException(500, detail.encode("ascii", "backslashreplace").decode("ascii"))


# ── v1.0: APA 7th Edition Word export ────────────────────────────────────────

@app.post("/export/docx")
async def export_docx_route(
    file: UploadFile = File(...),
    model: str = Form(...),
    algorithm: str = Form("pls"),
    bootstrap_n: int = Form(1000),
    missing: str = Form("listwise"),
    reverse_items: Optional[str] = Form(None),
):
    """
    Generate an APA 7th-edition Word (.docx) report for a PLS/CB/WLS model.

    Form fields
    -----------
    file         : CSV, XLSX, or SAV data file.
    model        : lavaan-style model syntax (=~ / ~ / ~~).
    algorithm    : ``pls`` (default) | ``cb`` | ``wls``.
    bootstrap_n  : Bootstrap replications for path CIs (default 1 000).

    Returns
    -------
    Streaming .docx download (naval_sem_report.docx).

    Produces four APA-formatted tables:
      1. Measurement model (loadings, AVE, CR, α)
      2. Discriminant validity (HTMT + √AVE diagonal)
      3. Structural model (β, t, p, CI, f², R²)
      (Table 4 — Indirect effects — requires a separate /indirect call
       and is not included in this single-step export.)

    Requires: pip install python-docx
    """
    try:
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(
                400,
                f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.",
            )

        bootstrap_n = min(bootstrap_n, 20_000)

        # ── Parse upload ─────────────────────────────────────────────────
        raw = await file.read()
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /export/docx: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file.")

        # ── Missing data + reverse scoring (parity with /run) ────────────
        if missing == "listwise":
            df = df.dropna()
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        df = auto_reverse_score(df, model, reverse_items=reverse_items)

        # ── Fit model ────────────────────────────────────────────────────
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: fit_model(
                    df, model,
                    algorithm=algorithm,
                    bootstrap_n=bootstrap_n,
                ),
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            logger.error("Model fit error in /export/docx: %s", exc, exc_info=True)
            raise HTTPException(500, "Model fitting failed. Check server logs.")

        # ── Generate DOCX ────────────────────────────────────────────────
        try:
            from app.export_docx import generate_docx
        except Exception:
            try:
                import importlib.util, pathlib
                spec = importlib.util.spec_from_file_location(
                    "export_docx",
                    pathlib.Path(__file__).parent / "export_docx.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                generate_docx = mod.generate_docx
            except Exception as ie:
                logger.error("export_docx import failed: %s", ie, exc_info=True)
                raise HTTPException(
                    500,
                    f"export_docx module failed to load: {ie!r}. "
                    "Ensure python-docx is installed: pip install python-docx",
                ) from ie

        try:
            buf = await asyncio.get_running_loop().run_in_executor(
                None, lambda: generate_docx(result)
            )
        except Exception as exc:
            try:
                logger.error("DOCX generation error: %r", exc, exc_info=True)
            except Exception:
                logger.error("DOCX generation error (unprintable): %s", type(exc).__name__)
            try:
                detail = f"DOCX generation failed: {exc!r}"
            except Exception:
                detail = f"DOCX generation failed: {type(exc).__name__} (unprintable)"
            raise HTTPException(
                500,
                detail.encode("ascii", "backslashreplace").decode("ascii"),
            )

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buf,
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            headers={
                "Content-Disposition": 'attachment; filename="naval_sem_report.docx"'
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        try:
            logger.error("Unhandled /export/docx failure: %r", exc, exc_info=True)
        except Exception:
            logger.error("Unhandled /export/docx failure (unprintable): %s",
                         type(exc).__name__)
        try:
            detail = f"/export/docx failed unexpectedly: {type(exc).__name__}: {exc!r}"
        except Exception:
            detail = (
                f"/export/docx failed unexpectedly: "
                f"{type(exc).__name__} (unprintable message)"
            )
        raise HTTPException(
            500,
            detail.encode("ascii", "backslashreplace").decode("ascii"),
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
    reverse_items: Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n    = min(bootstrap_n, 20_000)
    n_permutations = min(n_permutations, 20_000)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

        raw = await file.read()
        log("step", f"MGA: parsing uploaded file: {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /mga: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if group_col not in df.columns:
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

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
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
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "MGA unexpected error — see server logs for details")
            logger.error("Unexpected error in /mga: %s", exc, exc_info=True)
            raise HTTPException(500, "MGA analysis failed. Check server logs.")

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
    reverse_items: Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

        raw = await file.read()
        log("step", f"HOC: parsing uploaded file: {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /hoc: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))
            log("info", "Missing data: mean imputation applied")

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

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
            raise HTTPException(
                422,
                "No higher-order constructs detected in model syntax. "
                "Use /run for standard (first-order) models.",
            )

        if hoc_method not in ("repeated_indicator", "two_stage"):
            raise HTTPException(
                400,
                f"Unknown hoc_method '{hoc_method}'. "
                "Use 'repeated_indicator' or 'two_stage'.",
            )

        try:
            fn = (
                fit_hoc_repeated_indicator
                if hoc_method == "repeated_indicator"
                else fit_hoc_two_stage
            )
            result = await asyncio.get_running_loop().run_in_executor(
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
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "HOC unexpected error — see server logs for details")
            logger.error("Unexpected error in /hoc: %s", exc, exc_info=True)
            raise HTTPException(500, "HOC analysis failed. Check server logs.")

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
    reverse_items: Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

        raw = await file.read()
        log("step", f"Moderation: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /moderation: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

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
                raise HTTPException(422, f"Manifest moderation failed: {exc}")
            from app.schemas import ModerationResult, ModerationTerm, FitIndices, PathParameter
            terms  = [ModerationTerm(**t)   for t in _manifest_result["moderation_terms"]]
            params = [PathParameter(**p)    for p in _manifest_result["parameters"]]
            return ModerationResult(
                algorithm=_manifest_result["algorithm"],
                n_obs=_manifest_result["n_obs"],
                bootstrap_n=_manifest_result["bootstrap_n"],
                moderation_terms=terms,
                parameters=params,
                fit=FitIndices(**_manifest_result["fit"]),
                warnings=_manifest_result.get("warnings", []),
            )

        try:
            result = await asyncio.get_running_loop().run_in_executor(
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
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "Moderation unexpected error — see server logs for details")
            logger.error("Unexpected error in /moderation: %s", exc, exc_info=True)
            raise HTTPException(500, "Moderation analysis failed. Check server logs.")

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
    reverse_items: Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

        raw = await file.read()
        log("step", f"IPMA: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /ipma: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
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
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "IPMA unexpected error — see server logs for details")
            logger.error("Unexpected error in /ipma: %s", exc, exc_info=True)
            raise HTTPException(500, "IPMA analysis failed. Check server logs.")

        return result


# ── v0.7: NCA ─────────────────────────────────────────────────────────────────

@app.post("/nca", response_model=NCAResult)
async def nca_analysis(
    file:            UploadFile = File(...),
    model:           str        = Form(...),
    n_permutations:  int        = Form(1000),
    missing:         str        = Form("listwise"),
    run_id:          str        = Form(None),
    reverse_items:   Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    n_permutations = min(n_permutations, 20_000)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"NCA: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /nca: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_nca(
                    df, model,
                    n_permutations=n_permutations,
                    log_fn=log,
                ),
            )
        except ValueError as exc:
            log("error", f"NCA failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "NCA unexpected error — see server logs for details")
            logger.error("Unexpected error in /nca: %s", exc, exc_info=True)
            raise HTTPException(500, "NCA analysis failed. Check server logs.")

        return result


# ── v0.9: NCA-ESSE ────────────────────────────────────────────────────────────

@app.post("/nca-esse", response_model=NCAESSEResult)
async def nca_esse_analysis(
    file:              UploadFile = File(...),
    model:             str        = Form(...),
    n_permutations:    int        = Form(200),
    n_benchmark_reps:  int        = Form(200),
    seed:              int        = Form(42),
    missing:           str        = Form("listwise"),
    run_id:            str        = Form(None),
    reverse_items:     Optional[str] = Form(None),
):
    """
    NCA Effect Size Sensitivity Extension (NCA-ESSE).

    Becker, Richter, Ringle & Sarstedt (2026) — J. Bus. Res. 206, 115920.

    Sweeps an ECDF threshold p from 0–5 % in 0.5 pt steps, recomputing the
    CE-FDH ceiling at each step after discarding the most extreme ceiling-
    violating observations. The empirical sensitivity curve is compared
    against a joint-uniform benchmark (no necessity by construction) to
    identify the largest threshold where relaxing the ceiling still reflects
    genuine signal rather than chance. A permutation test (shuffled Y) is
    applied at every threshold.

    Form fields
    -----------
    file              : CSV, XLSX, or SAV dataset.
    model             : lavaan syntax. All structural IV → DV pairs are tested.
    n_permutations    : Permutation samples per threshold per pair (default 200).
    n_benchmark_reps  : Joint-uniform benchmark replications per pair (default 200).
    seed              : RNG seed for reproducibility (default 42).
    missing           : ``listwise`` (default) | ``mean``.
    run_id            : Optional SSE tracking ID.

    Returns
    -------
    NCAESSEResult — per-pair sensitivity curves, benchmark curves,
    recommended threshold and effect size, ceiling line coordinates,
    and permutation p-values at every threshold step.
    """
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    n_permutations    = min(n_permutations, 5_000)
    n_benchmark_reps  = min(n_benchmark_reps, 5_000)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"NCA-ESSE: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /nca-esse: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_nca_esse(
                    df, model,
                    n_permutations=n_permutations,
                    n_benchmark_reps=n_benchmark_reps,
                    seed=seed,
                    log_fn=log,
                ),
            )
        except ValueError as exc:
            log("error", f"NCA-ESSE failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "NCA-ESSE unexpected error — see server logs for details")
            logger.error("Unexpected error in /nca-esse: %s", exc, exc_info=True)
            raise HTTPException(500, "NCA-ESSE analysis failed. Check server logs.")

        return result


# ── v1.0: fsQCA (fuzzy-set Qualitative Comparative Analysis) ──────────────────

@app.post("/fsqca", response_model=FsQCAResult)
async def run_fsqca_endpoint(
    file:              UploadFile = File(...),
    outcome:           str        = Form(...),
    conditions:        str        = Form(...),   # comma-separated column names
    freq_threshold:    int        = Form(1),
    consist_threshold: float      = Form(0.75),
    missing:           str        = Form("listwise"),
):
    """
    Fuzzy-set Qualitative Comparative Analysis (fsQCA).

    Form fields
    -----------
    file              : CSV, XLSX, or SAV dataset.
                        Columns should already be calibrated to fuzzy membership
                        scores in (0, 1).  Values outside this range trigger
                        automatic indirect (percentile-based) calibration.
    outcome           : Column name of the outcome fuzzy set.
    conditions        : Comma-separated column names of the condition fuzzy sets.
    freq_threshold    : Minimum cases per truth-table row for the row to count
                        as non-remainder (default 1).
    consist_threshold : Minimum PRI consistency score for a truth-table row to be
                        coded outcome=1 (default 0.75).
    missing           : ``listwise`` (default) | ``mean``.

    Returns
    -------
    FsQCAResult — necessity analysis, truth table, three minimized solutions
    (complex / parsimonious / intermediate), and XY bubble-chart data.
    """
    raw = await file.read()
    try:
        df = _parse_upload(raw, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File parse error in /fsqca: %s", exc, exc_info=True)
        raise HTTPException(422, "Could not parse the uploaded file.")

    if missing == "listwise":
        df = df.dropna()
    elif missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))

    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    if not cond_list:
        raise HTTPException(400, "At least one condition column must be provided.")

    from app.fsqca import run_fsqca
    try:
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_fsqca(
                df,
                outcome,
                cond_list,
                calibration_params={},
                freq_threshold=freq_threshold,
                consist_threshold=consist_threshold,
            ),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /fsqca: %s", exc, exc_info=True)
        raise HTTPException(500, "fsQCA analysis failed. Check server logs.")


# ── v0.8: Robustness Checks ───────────────────────────────────────────────────

@app.post("/robustness", response_model=RobustnessChecks)
async def run_robustness(
    model:        str   = Form(...),
    data:         UploadFile = File(...),
    algorithm:    str   = Form("pls"),
    checks:       str   = Form("nonlinear,copula"),
    endogenous:   str   = Form(""),          # comma-sep vars for copula
    bootstrap_n:  int   = Form(500),
    seed:         int   = Form(42),
    scale_min:    float = Form(None),
    scale_max:    float = Form(None),
    run_id:       str   = Form(""),
):
    if algorithm not in ("pls", "cb", "wls"):
        raise HTTPException(400, f"Invalid algorithm '{algorithm}'.")
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log_fn = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await data.read()
        df = _parse_upload(raw, data.filename)
        model = _expand_covariances(model)
        checks_set = {c.strip() for c in checks.split(",") if c.strip()}
        endogenous_list = [v.strip() for v in endogenous.split(",") if v.strip()]

        result = RobustnessChecks()
        loop = asyncio.get_running_loop()

        if "nonlinear" in checks_set:
            from app.engine import compute_nonlinear_effects
            result.nonlinear = await loop.run_in_executor(None, lambda: compute_nonlinear_effects(
                df, model, algorithm=algorithm, bootstrap_n=bootstrap_n, seed=seed, log_fn=log_fn))

        if "copula" in checks_set and endogenous_list:
            from app.engine import compute_gaussian_copula
            try:
                result.copula = await loop.run_in_executor(None, lambda: compute_gaussian_copula(
                    df, model, endogenous_list, algorithm=algorithm, bootstrap_n=bootstrap_n, seed=seed, log_fn=log_fn))
            except ValueError as _cop_exc:
                result.copula = None
                result.copula_warning = str(_cop_exc)
                log_fn("warn", f"Gaussian Copula failed: {_cop_exc}")
            except Exception as _cop_exc:
                logger.error("Unexpected error in copula check: %s", _cop_exc, exc_info=True)
                log_fn("warn", f"Gaussian Copula check failed unexpectedly: {_cop_exc}")
                result.copula = None
        elif "copula" in checks_set and not endogenous_list:
            log_fn("warn", "Copula check requested but no endogenous variables specified — skipped. Pass endogenous=VAR1,VAR2.")

        return result


# ── v0.8: FIMIX-PLS ───────────────────────────────────────────────────────────

@app.post("/fimix", response_model=FIMIXResult)
async def run_fimix_endpoint(
    model:       str  = Form(...),
    data:        UploadFile = File(...),
    k_max:       int  = Form(5),
    n_starts:    int  = Form(10),
    bootstrap_n: int  = Form(0),
    seed:        int  = Form(42),
    run_id:      str  = Form(""),
):
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    log_fn = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await data.read()
        df = _parse_upload(raw, data.filename)
        from app.fimix import run_fimix
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: run_fimix(
            df, model, k_max=k_max, n_starts=n_starts, seed=seed, log_fn=log_fn))
        return result


# ── v0.8: PLS-POS ─────────────────────────────────────────────────────────────

@app.post("/plspos", response_model=PLSPOSResult)
async def run_plspos_endpoint(
    model:              str  = Form(...),
    data:               UploadFile = File(...),
    k:                  int  = Form(...),
    n_starts:           int  = Form(10),
    seed:               int  = Form(42),
    fimix_result_json:  str  = Form(""),   # optional: JSON string of a prior FIMIXResult
    run_id:             str  = Form(""),
):
    run_id = run_id or str(uuid.uuid4())
    _init_run(run_id)
    if k < 2:
        raise HTTPException(status_code=422, detail=f"k must be ≥ 2 for PLS-POS (got k={k}). Use FIMIX to determine an appropriate number of segments.")
    log_fn = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await data.read()
        df = _parse_upload(raw, data.filename)

        # Deserialise the prior FIMIX result when provided so run_plspos()
        # can (a) warm-start from FIMIX segment memberships and (b) populate
        # the fimix_comparison table in PLSPOSResult.
        fimix_result = None
        if fimix_result_json.strip():
            try:
                fimix_result = FIMIXResult.model_validate_json(fimix_result_json)
            except Exception as exc:
                raise HTTPException(400, f"Invalid fimix_result_json: {exc}")

        from app.plspos import run_plspos
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, lambda: run_plspos(
                df, model, k=k, fimix_result=fimix_result,
                n_starts=n_starts, seed=seed, log_fn=log_fn))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            logger.error("Unexpected error in /plspos: %s", exc, exc_info=True)
            raise HTTPException(500, "PLS-POS analysis failed. Check server logs.")
        return result

@app.post("/mod-mediation", response_model=ModMediationResult)
async def mod_mediation_analysis(
    file:        UploadFile = File(...),
    model:       str        = Form(...),
    algorithm:   str        = Form("pls"),
    bootstrap_n: int        = Form(500),
    missing:     str        = Form("listwise"),
    run_id:      str        = Form(None),
    reverse_items: Optional[str] = Form(None),
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
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    bootstrap_n = min(bootstrap_n, 20_000)
    with _run_context(run_id):
        if algorithm not in ("pls", "cb", "wls"):
            raise HTTPException(400, f"Invalid algorithm '{algorithm}'. Use 'pls', 'cb', or 'wls'.")

        raw = await file.read()
        log("step", f"ModMediation: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /mod-mediation: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file. Ensure it is a valid CSV, XLSX, or SAV.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        df = auto_reverse_score(df, model, log_fn=log, reverse_items=reverse_items)

        try:
            result = await asyncio.get_running_loop().run_in_executor(
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
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "ModMediation unexpected error — see server logs for details")
            logger.error("Unexpected error in /mod-mediation: %s", exc, exc_info=True)
            raise HTTPException(500, "Moderated mediation analysis failed. Check server logs.")

        return result


# ── v0.9: Nomological Validity ─────────────────────────────────────────────────

@app.post("/nomological", response_model=List[NomologicalResult])
async def run_nomological(
    file: UploadFile = File(...),
    model_syntax: str = Form(...),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"Nomological: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /nomological: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file.")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_nomological_validity(df, model_syntax),
            )
        except ValueError as exc:
            log("error", f"Nomological failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "Nomological unexpected error — see server logs")
            logger.error("Unexpected error in /nomological: %s", exc, exc_info=True)
            raise HTTPException(500, "Nomological validity analysis failed.")

        log("ok", f"Nomological complete — {len(result)} construct(s)")
        return result


# ── v0.9: Measurement Invariance ───────────────────────────────────────────────

@app.post("/invariance", response_model=MeasurementInvarianceResult)
async def run_invariance(
    file: UploadFile = File(...),
    model_syntax: str = Form(...),
    group_col: str = Form(...),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    """
    Full measurement invariance sequence: configural → metric → scalar.

    Parameters
    ----------
    file         : CSV upload containing all variables + the group column.
    model_syntax : lavaan-style model syntax (=~ / ~ / ~~).
    group_col    : Column name whose values define groups (≥ 2 distinct values).
    missing      : ``listwise`` (default) | ``mean``.
    run_id       : Optional SSE tracking ID — logs available at /logs/{run_id}.

    Returns
    -------
    MeasurementInvarianceResult — per-level CFI / RMSEA / SRMR / ΔCFI / ΔRMSEA,
    partial invariance items (if scalar partially holds), and a plain-English
    conclusion: "Full scalar" / "Partial scalar" / "Metric only" / "Configural only".
    """
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"Invariance: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /invariance: %s", exc, exc_info=True)
            raise HTTPException(422, f"Could not parse uploaded file: {exc}")

        if missing == "listwise":
            df = df.dropna()
            log("info", f"Missing data: listwise deletion → {len(df)} complete rows")
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_measurement_invariance(df, model_syntax, group_col),
            )
        except ValueError as exc:
            log("error", f"Invariance failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "Invariance unexpected error — see server logs")
            logger.error("Unexpected error in /invariance: %s", exc, exc_info=True)
            raise HTTPException(500, "Measurement invariance analysis failed. Check server logs.")

        log("ok", f"Invariance complete — conclusion: {result.conclusion}")
        return result


# ── v0.9: EFA (Exploratory Factor Analysis) ────────────────────────────────────

@app.post("/efa", response_model=ScaleDevelopmentResult)
async def run_efa(
    file: UploadFile = File(...),
    n_factors: Optional[int] = Form(None),
    rotation: str = Form("varimax"),
    missing: str = Form("listwise"),
    run_id: str = Form(None),
):
    """
    Exploratory Factor Analysis with KMO, Bartlett's test, and varimax/oblimin rotation.

    Form fields
    -----------
    file      : CSV file — rows = respondents, columns = items (numeric).
    n_factors : Number of factors to extract; if omitted, Kaiser criterion (λ > 1) is used.
    rotation  : Rotation method passed to sklearn FactorAnalysis (default ``varimax``).
    missing   : ``listwise`` (default) | ``mean``.
    run_id    : Optional SSE tracking ID — logs available at /logs/{run_id}.

    Returns
    -------
    ScaleDevelopmentResult — KMO, Bartlett χ², eigenvalues, variance explained,
    factor loadings, cross-loadings, and warnings.
    """
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"EFA: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /efa: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file.")

        if missing == "listwise":
            df = df.dropna()
        elif missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_efa(df, n_factors=n_factors, rotation=rotation, log_fn=log),
            )
        except ValueError as exc:
            log("error", f"EFA failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "EFA unexpected error — see server logs")
            logger.error("Unexpected error in /efa: %s", exc, exc_info=True)
            raise HTTPException(500, "EFA analysis failed. Check server logs.")

        log("ok", f"EFA complete — {result.n_factors} factor(s), KMO={result.kmo}")
        return result


# ── v0.9: CVI (Content Validity Index) ────────────────────────────────────────

@app.post("/cvi", response_model=CVIResult)
async def run_cvi(
    file: UploadFile = File(...),
    n_experts: int = Form(...),
    run_id: str = Form(None),
):
    """
    Content Validity Index from an expert ratings matrix.

    Form fields
    -----------
    file      : CSV file — rows = experts, columns = items, values = 1–4 Likert ratings.
    n_experts : Number of experts (rows) used for I-CVI proportion denominators.
    run_id    : Optional SSE tracking ID — logs available at /logs/{run_id}.

    Returns
    -------
    CVIResult — I-CVI per item, S-CVI/Ave, S-CVI/UA, modified kappa (κ*),
    and an interpretation (Excellent / Acceptable / Poor).
    """
    run_id = run_id or str(uuid.uuid4())
    _validate_run_id(run_id)
    _init_run(run_id)
    log = _make_log_fn(run_id)
    with _run_context(run_id):
        raw = await file.read()
        log("step", f"CVI: parsing {file.filename}")
        try:
            df = _parse_upload(raw, file.filename)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("File parse error in /cvi: %s", exc, exc_info=True)
            raise HTTPException(422, "Could not parse the uploaded file.")

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: compute_cvi(df, n_experts=n_experts),
            )
        except ValueError as exc:
            log("error", f"CVI failed: {exc}")
            raise HTTPException(422, str(exc))
        except Exception as exc:
            log("error", "CVI unexpected error — see server logs")
            logger.error("Unexpected error in /cvi: %s", exc, exc_info=True)
            raise HTTPException(500, "CVI analysis failed. Check server logs.")

        log("ok", f"CVI complete — {result.n_items} item(s), interpretation: {result.interpretation}")
        return result


# ── Static files — MUST be registered last ────────────────────────────────────
# Starlette evaluates routes in insertion order. Mounting StaticFiles at "/"
# before any API route would shadow every endpoint. Mounting here ensures all
# @app.get / @app.post routes are resolved first; only unmatched paths fall
# through to the static file handler.
#
# html=True means StaticFiles will serve index.html for bare directory requests
# (e.g. GET /), so the explicit @app.get("/") route is no longer needed.
if Path(_STATIC_DIR).exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
else:
    @app.get("/", include_in_schema=False)
    def root_missing():
        return JSONResponse(
            {"error": "Static files not found. Check server configuration."},
            status_code=500,
        )
