"""
scripts/test_dashboard_routes.py

Purpose: check the dashboard page end to end (login guard, empty state, real
numbers, per-user isolation, HTML escaping, the "latest 10" limit) using
FastAPI's TestClient and a THROWAWAY database. Your real database is untouched.

Run from the project root, inside your fastapi-env:
    python scripts/test_dashboard_routes.py
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
sys.path.insert(0, str(ROOT / "src" / "storage"))
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth_routes
import dashboard_routes
import db_store
from db_core import engine

db_store.init_db()

app = FastAPI()
app.include_router(auth_routes.router)
app.include_router(dashboard_routes.router)

NO_FOLLOW = {"follow_redirects": False}


def scored_df(label="squats", reps=9.0, quality=95.0,
              feedback="Good form and consistency.") -> pd.DataFrame:
    """One exercise set followed by a rest set (which must never be shown)."""
    return pd.DataFrame({
        "label": [label, "non_activity"],
        "start_time": [1_700_000_000.0, 1_700_000_030.0],
        "end_time": [1_700_000_030.0, 1_700_000_045.0],
        "duration_s": [30.0, 15.0],
        "num_windows": [10, 5],
        "mean_confidence": [0.9, 0.8],
        "confidence_std": [0.05, float("nan")],
        "raw_agreement": [0.95, float("nan")],
        "is_short": [False, False],
        "estimated_reps": [reps, float("nan")],
        "quality_score": [quality, float("nan")],
        "feedback": [feedback, None],
    })


def register(client, username):
    r = client.post("/register", data={"username": username, "password": "password123",
                                       "confirm": "password123"}, **NO_FOLLOW)
    assert r.status_code == 303


# --- anonymous visitors are sent to login -------------------------------------------
anon = TestClient(app)
r = anon.get("/", **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/login"
print("guard: OK")

# --- a brand-new user sees the empty state --------------------------------------------
alice_client = TestClient(app)
register(alice_client, "alice")
r = alice_client.get("/")
assert r.status_code == 200
assert "No sessions yet" in r.text and "Logged in as <b>alice</b>" in r.text
assert 'href="/capture"' in r.text and 'action="/logout"' in r.text
assert r.headers["cache-control"] == "no-store"
assert '<span class="num">0</span><span class="lbl">Sessions</span>' in r.text
assert '<span class="num">-</span><span class="lbl">Avg quality</span>' in r.text
print("empty state: OK")

# --- real numbers ----------------------------------------------------------------------
alice_id = alice_client.get("/api/me").json()["id"]
db_store.save_session(scored_df("squats", 9.0, 95.0), "s1", "Squats_morning", alice_id)
db_store.save_session(scored_df("pushups", 7.0, 55.0, "Movement pattern unstable."),
                      "s2", "<script>alert(1)</script>", alice_id)

r = alice_client.get("/")
page = r.text
assert "No sessions yet" not in page
assert '<span class="num">2</span><span class="lbl">Sessions</span>' in page
assert '<span class="num">16</span><span class="lbl">Total reps</span>' in page      # 9 + 7
assert '<span class="num">75</span><span class="lbl">Avg quality</span>' in page     # (95+55)/2
assert page.count('<details class="session">') == 2
assert "Squats &times;9" in page and "Pushups &times;7" in page
assert "Good form and consistency." in page and "Movement pattern unstable." in page
assert "q-good" in page and "q-low" in page                                          # 95 and 55
assert "Non Activity" not in page and "non_activity" not in page                     # rest hidden
assert page.index("Squats_morning") > page.index("&lt;script&gt;")                   # newest first
assert "<script>alert(1)</script>" not in page                                       # escaped
assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
assert '<time datetime="' in page
print("numbers + sessions: OK")

# --- another user sees none of it --------------------------------------------------------
bob_client = TestClient(app)
register(bob_client, "bob")
bob_page = bob_client.get("/").text
assert "No sessions yet" in bob_page
assert "Squats_morning" not in bob_page and "Good form" not in bob_page
assert '<span class="num">0</span><span class="lbl">Total reps</span>' in bob_page
print("per-user isolation: OK")

# --- only the latest 10 are listed ------------------------------------------------------------
for i in range(3, 13):                                        # s3 ... s12 (10 more)
    db_store.save_session(scored_df(), f"s{i}", f"Session_{i}", alice_id)
page = alice_client.get("/").text
assert page.count('<details class="session">') == 10
assert "Showing your latest 10 of 12 sessions" in page
assert '<span class="num">12</span><span class="lbl">Sessions</span>' in page
print("latest-10 limit: OK")

print("\nAll dashboard checks passed.")
engine.dispose()
