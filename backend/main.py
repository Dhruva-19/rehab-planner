"""
FastAPI backend - Day 14 (pipeline-wired)

Receives sensor buffers captured in-browser and runs them through the
SAME pipeline app.py uses:
    write CSVs (matching sensorlogger_to_upload_csv.py's schema)
    -> predict_from_raw_csv   (windows + predictions)
    -> aggregate_into_sets    (windows -> exercise sets + rep counts)
    -> score_sets             (quality scoring)
    -> save_session           (persist to the same SQLite DB app.py reads,
                                so live-captured sessions show up under
                                Streamlit's "Past Sessions" tab too)

The model bundle is loaded ONCE at server startup (not per-request) -
predict_from_raw_csv accepts a pre-loaded bundle specifically to support
this, avoiding a slow pickle/npz reload on every single recording.

Run with (from the backend/ folder):
    python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Make sibling src/ packages importable, same pattern as app.py -----
# main.py lives in backend/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src" / "inference"))
sys.path.append(str(PROJECT_ROOT / "src" / "feedback"))
sys.path.append(str(PROJECT_ROOT / "src" / "storage"))

from predict_pipeline import predict_from_raw_csv, load_model_bundle, MODEL_PATH, WINDOWS_NPZ_PATH  # noqa: E402
from aggregate_sets import aggregate_into_sets  # noqa: E402
from quality_scorer import score_sets, session_summary  # noqa: E402
from db import init_db, migrate_add_scoring_columns, save_session  # noqa: E402

# Same convention as sensorlogger_to_upload_csv.py: data/sample_upload,
# anchored to PROJECT_ROOT so it resolves to the same folder regardless
# of which directory uvicorn is launched from.
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample_upload"

# Holds the model bundle once loaded - avoids reloading on every request.
_model_bundle: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: runs once when the server boots ---
    global _model_bundle
    init_db()
    migrate_add_scoring_columns()
    print("Loading model bundle...")
    resolved_model_path = PROJECT_ROOT / MODEL_PATH
    resolved_windows_npz_path = PROJECT_ROOT / WINDOWS_NPZ_PATH
    _model_bundle = load_model_bundle(str(resolved_model_path), str(resolved_windows_npz_path))
    print("Model bundle loaded. Backend ready.")
    yield
    # --- Shutdown: nothing to clean up currently ---


app = FastAPI(title="Rehab Planner Ingest API", lifespan=lifespan)
from fastapi.responses import FileResponse

@app.get("/")
def serve_test_page():
    html_path = PROJECT_ROOT / "sensor_test" / "sensor_capture_test.html"
    return FileResponse(html_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorSample(BaseModel):
    t: float
    x: float
    y: float
    z: float


class SessionPayload(BaseModel):
    session_name: Optional[str] = None
    accel: List[SensorSample]
    gyro: List[SensorSample]


def sanitize_session_name(raw: Optional[str]) -> str:
    if raw and raw.strip():
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip())
        return cleaned
    return f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def samples_to_dataframe(samples: List[SensorSample]) -> pd.DataFrame:
    return pd.DataFrame({
        "frame": 0,
        "timestamp_ms": [s.t for s in samples],
        "x": [s.x for s in samples],
        "y": [s.y for s in samples],
        "z": [s.z for s in samples],
    })


def sanitize_for_json(obj):
    """Recursively replace NaN/inf with None so nothing here can ever break
    Starlette's strict JSON encoder (which rejects NaN outright, unlike
    Python's json.dumps which tolerates it by default). Handles dicts,
    lists, and numpy/pandas scalar types - NaN can hide in either the
    per-set records (e.g. quality_score) or the summary dict (e.g.
    avg_quality_score, which is NaN when zero sets were scored)."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def dataframe_to_json_safe(df: pd.DataFrame) -> list:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return sanitize_for_json(records)


@app.post("/ingest")
def ingest_session(payload: SessionPayload):
    if not payload.accel or not payload.gyro:
        return {"status": "error", "message": "Both accel and gyro buffers are required and must be non-empty."}

    session_name = sanitize_session_name(payload.session_name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    acc_path = OUTPUT_DIR / f"{session_name}_acc.csv"
    gyro_path = OUTPUT_DIR / f"{session_name}_gyro.csv"

    acc_df = samples_to_dataframe(payload.accel)
    gyro_df = samples_to_dataframe(payload.gyro)
    acc_df.to_csv(acc_path, header=False, index=False)
    gyro_df.to_csv(gyro_path, header=False, index=False)

    # --- Run the same pipeline app.py's New Session tab runs ---------
    try:
        window_results = predict_from_raw_csv(str(acc_path), str(gyro_path), bundle=_model_bundle)

        # Raw gyro for rep-counting, same read pattern as app.py
        raw_gyro_ts = (gyro_df["timestamp_ms"] / 1000.0).to_numpy()
        raw_gyro_xyz = gyro_df[["x", "y", "z"]].to_numpy()

        session_sets = aggregate_into_sets(
            window_results,
            raw_gyro_ts=raw_gyro_ts,
            raw_gyro_xyz=raw_gyro_xyz,
        )
        scored = score_sets(session_sets)
        summary = session_summary(scored)

        session_id = f"{session_name}_{datetime.now():%Y%m%d_%H%M%S}"
        save_session(scored, session_id=session_id, source_name=session_name)

    except ValueError as e:
        # e.g. "not enough data for one window" - a legitimate, expected
        # failure mode (recording too short), not a bug - report it cleanly.
        print(f"INGEST PIPELINE ERROR (session '{session_name}'): {e}")
        return {
            "status": "error",
            "message": str(e),
            "accel_samples": len(payload.accel),
            "gyro_samples": len(payload.gyro),
        }

    return {
        "status": "ok",
        "session_id": session_id,
        "accel_samples": len(payload.accel),
        "gyro_samples": len(payload.gyro),
        "summary": sanitize_for_json(summary),
        "sets": dataframe_to_json_safe(scored),
    }


@app.get("/health")
def health():
    return {"status": "backend is running", "model_loaded": _model_bundle is not None}