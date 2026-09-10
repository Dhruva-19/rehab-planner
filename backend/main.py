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

from fastapi import FastAPI, Form, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse
import uuid
import sqlite3
from datetime import datetime, timedelta
from src.storage.db import create_user, verify_user, get_connection, migrate_add_users_table_and_user_id

migrate_add_users_table_and_user_id()  # safe to call repeatedly, uses IF NOT EXISTS

# In-memory session store: {session_token: user_id}. Fine for a single-instance
# Render demo — resets on restart, same tradeoff as the ephemeral filesystem.
SESSIONS = {}

def get_current_user(session_token: str = Cookie(default=None)):
    if session_token and session_token in SESSIONS:
        return SESSIONS[session_token]
    return None

# ---------- Extra tables: goals ----------
def ensure_goals_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

ensure_goals_table()

# ---------- Auth pages ----------
@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <h2>Rehab Planner Login</h2>
    <form method="post" action="/login">
      <input name="username" placeholder="Username"><br>
      <input name="password" type="password" placeholder="Password"><br>
      <button type="submit">Log In</button>
    </form>
    <p><a href="/signup">Sign up instead</a></p>
    """

@app.post("/login")
def login_submit(username: str = Form(...), password: str = Form(...)):
    user_id = verify_user(username, password)
    if not user_id:
        return HTMLResponse("<p>Invalid login. <a href='/login'>Try again</a></p>")
    token = str(uuid.uuid4())
    SESSIONS[token] = user_id
    resp = RedirectResponse(url="/dashboard", status_code=302)
    resp.set_cookie(key="session_token", value=token)
    return resp

@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return """
    <h2>Sign Up</h2>
    <form method="post" action="/signup">
      <input name="username" placeholder="Username"><br>
      <input name="password" type="password" placeholder="Password"><br>
      <button type="submit">Sign Up</button>
    </form>
    """

@app.post("/signup")
def signup_submit(username: str = Form(...), password: str = Form(...)):
    if create_user(username, password):
        return RedirectResponse(url="/login", status_code=302)
    return HTMLResponse("<p>Username taken. <a href='/signup'>Try again</a></p>")

@app.get("/logout")
def logout(session_token: str = Cookie(default=None)):
    SESSIONS.pop(session_token, None)
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("session_token")
    return resp

# ---------- Dashboard ----------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(session_token: str = Cookie(default=None)):
    user_id = get_current_user(session_token)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    # --- /dashboard ---
    rows = conn.execute(
        "SELECT sessions.uploaded_at, sets.label, sets.quality_score "
        "FROM sets JOIN sessions ON sets.session_id = sessions.session_id "
        "WHERE sessions.user_id = ? OR sessions.user_id IS NULL "
        "ORDER BY sessions.uploaded_at DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    conn.close()

    rows_html = "".join(
        f"<tr><td>{r[1]}</td><td>{r[2]}</td><td>{r[3] if r[3] is not None else '-'}</td></tr>"
        for r in rows
    )
    return f"""
    <h2>Dashboard</h2>
    <p><a href="/progress">Progress Trend</a> | <a href="/streak">Streak</a> | 
       <a href="/goals">Goals</a> | <a href="/logout">Log out</a></p>
    <table border="1" cellpadding="6">
      <tr><th>Date</th><th>Exercise</th><th>Quality Score</th></tr>
      {rows_html}
    </table>
    """

# ---------- Progress trend ----------
@app.get("/progress", response_class=HTMLResponse)
def progress(session_token: str = Cookie(default=None)):
    user_id = get_current_user(session_token)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    # --- /progress ---
    rows = conn.execute(
        "SELECT substr(sessions.uploaded_at, 1, 10) AS day, AVG(sets.quality_score) "
        "FROM sets JOIN sessions ON sets.session_id = sessions.session_id "
        "WHERE (sessions.user_id = ? OR sessions.user_id IS NULL) AND sets.quality_score IS NOT NULL "
        "GROUP BY day ORDER BY day",
        (user_id,)
    ).fetchall()
    conn.close()

    rows_html = "".join(f"<tr><td>{r[0]}</td><td>{r[1]:.1f}</td></tr>" for r in rows)
    return f"""
    <h2>Progress Trend (avg quality score per session)</h2>
    <p><a href="/dashboard">Back</a></p>
    <table border="1" cellpadding="6">
      <tr><th>Date</th><th>Avg Quality</th></tr>
      {rows_html}
    </table>
    """

# ---------- Streak tracker ----------
@app.get("/streak", response_class=HTMLResponse)
def streak(session_token: str = Cookie(default=None)):
    user_id = get_current_user(session_token)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
   # --- /streak ---
    rows = conn.execute(
        "SELECT DISTINCT substr(uploaded_at, 1, 10) AS day FROM sessions "
        "WHERE user_id = ? OR user_id IS NULL ORDER BY day DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    dates = [datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows if r[0]]
    streak_count = 0
    if dates:
        expected = dates[0]
        for d in dates:
            if d == expected:
                streak_count += 1
                expected = expected - timedelta(days=1)
            else:
                break

    return f"""
    <h2>Current Streak: {streak_count} day(s)</h2>
    <p><a href="/dashboard">Back</a></p>
    """

# ---------- Goal setting ----------
@app.get("/goals", response_class=HTMLResponse)
def goals_page(session_token: str = Cookie(default=None)):
    user_id = get_current_user(session_token)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    rows = conn.execute(
        "SELECT goal_text, created_at FROM goals WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    goals_html = "".join(f"<li>{r[0]} <small>({r[1]})</small></li>" for r in rows)
    return f"""
    <h2>Your Goals</h2>
    <p><a href="/dashboard">Back</a></p>
    <form method="post" action="/goals">
      <input name="goal_text" placeholder="e.g. 3 sessions this week"><br>
      <button type="submit">Add Goal</button>
    </form>
    <ul>{goals_html}</ul>
    """

@app.post("/goals")
def add_goal(goal_text: str = Form(...), session_token: str = Cookie(default=None)):
    user_id = get_current_user(session_token)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    conn = get_connection()
    conn.execute(
        "INSERT INTO goals (user_id, goal_text, created_at) VALUES (?, ?, ?)",
        (user_id, goal_text, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()
    return RedirectResponse(url="/goals", status_code=302)

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