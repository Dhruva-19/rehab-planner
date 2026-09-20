"""
scripts/test_report_routes.py

Purpose: check the PDF report download end to end -- login guard, ownership
(another user's session is a plain 404), the PDF response headers, awkward
text (markup characters in a session name), a rest-only session (no average
score), and the small fixes made to pdf_report.py. Uses a THROWAWAY database.

Needs the `reportlab` package (the report library) in your fastapi-env.

Run from the project root, inside your fastapi-env:
    python scripts/test_report_routes.py
"""

import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_tmp.name) / 'test.db').as_posix()}"
os.environ.pop("RENDER", None)
os.environ.pop("COOKIE_SECURE", None)

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/storage", "src/feedback", "src/reporting", "backend"):
    sys.path.insert(0, str(ROOT / sub))

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reportlab.lib.styles import ParagraphStyle

import auth_routes
import db_store
import pdf_report
import report_routes
from db_core import engine

db_store.init_db()

app = FastAPI()
app.include_router(auth_routes.router)
app.include_router(report_routes.router)

NO_FOLLOW = {"follow_redirects": False}


def scored_df(rest_only=False) -> pd.DataFrame:
    exercise = {
        "label": "squats", "start_time": 1_700_000_000.0, "end_time": 1_700_000_030.0,
        "duration_s": 30.0, "num_windows": 10, "mean_confidence": 0.9,
        "confidence_std": 0.05, "raw_agreement": 0.95, "is_short": False,
        "estimated_reps": 9.0, "quality_score": 95.0,
        "feedback": "Good form and consistency.",
    }
    rest = {
        "label": "non_activity", "start_time": 1_700_000_030.0, "end_time": 1_700_000_045.0,
        "duration_s": 15.0, "num_windows": 5, "mean_confidence": 0.8,
        "confidence_std": float("nan"), "raw_agreement": float("nan"), "is_short": False,
        "estimated_reps": float("nan"), "quality_score": float("nan"),
        "feedback": "Not scored (rest period).",
    }
    return pd.DataFrame([rest] if rest_only else [exercise, rest])


def register(client, username):
    r = client.post("/register", data={"username": username, "password": "password123",
                                       "confirm": "password123"}, **NO_FOLLOW)
    assert r.status_code == 303


# --- pdf_report.py fixes, checked directly ------------------------------------------------
style = ParagraphStyle("t")
assert pdf_report._format_cell("feedback", None, style) == "-"                    # NULL -> "-"
assert pdf_report._format_cell("feedback", float("nan"), style) == "-"
assert pdf_report._format_cell("feedback", "a < b & c", style).text == "a &lt; b &amp; c"
assert pdf_report._format_cell("quality_score", float("nan"), style) == "-"
print("pdf_report hardening: OK")

# --- guard -------------------------------------------------------------------------------------
anon = TestClient(app)
r = anon.get("/sessions/whatever/report.pdf", **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/login"
print("guard: OK")

# --- a normal report -----------------------------------------------------------------------------
alice = TestClient(app)
register(alice, "alice")
alice_id = alice.get("/api/me").json()["id"]
db_store.save_session(scored_df(), "s1", "Squats_morning", alice_id)

r = alice.get("/sessions/s1/report.pdf")
assert r.status_code == 200
assert r.headers["content-type"] == "application/pdf"
assert r.content.startswith(b"%PDF-") and len(r.content) > 1500
assert r.headers["content-disposition"] == 'attachment; filename="rehab_report_Squats_morning.pdf"'
assert r.headers["cache-control"] == "private, no-store"
print("normal report: OK")

# --- ownership: someone else's session and an unknown id look identical -------------------------
bob = TestClient(app)
register(bob, "bob")
theirs = bob.get("/sessions/s1/report.pdf")
unknown = bob.get("/sessions/does-not-exist/report.pdf")
assert theirs.status_code == 404 and unknown.status_code == 404
assert theirs.text == unknown.text == "Session not found."
print("ownership: OK")

# --- awkward session name: markup characters must not break or hijack the PDF --------------------
nasty = '<b>bold</b> & <img src="file:///etc/passwd"> <unclosed'
db_store.save_session(scored_df(), "s2", nasty, alice_id)
r = alice.get("/sessions/s2/report.pdf")
assert r.status_code == 200 and r.content.startswith(b"%PDF-")
assert 'filename="rehab_report_b_bold_b_img_src_file_etc_passwd_unclosed.pdf"' in r.headers["content-disposition"]
print("markup in session name: OK")

# --- a rest-only session has no average score (NaN): must still produce a report -----------------------
db_store.save_session(scored_df(rest_only=True), "s3", "Rest_only", alice_id)
r = alice.get("/sessions/s3/report.pdf")
assert r.status_code == 200 and r.content.startswith(b"%PDF-")
print("rest-only session: OK")

print("\nAll report checks passed.")
engine.dispose()
