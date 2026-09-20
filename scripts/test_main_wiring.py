"""
scripts/test_main_wiring.py

Purpose: confirm that backend/main.py is wired to the login system correctly:
  * /health stays public
  * the dashboard (/) and the capture page (/capture) redirect anonymous
    visitors to /login
  * /ingest answers 401 to anonymous callers (before any ML work happens)
  * after registering, the dashboard and the capture page both load

It uses a THROWAWAY database and does not load the ML model (the server
"lifespan" startup is not run by a plain TestClient), so it is fast and never
touches your real rehab_planner_v2.db.

Run from the project root, inside your fastapi-env:
    python scripts/test_main_wiring.py
"""

import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_tmp.name) / 'test.db').as_posix()}"
os.environ.pop("RENDER", None)
os.environ.pop("COOKIE_SECURE", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient

import main                 # must come first: it puts src/storage on sys.path
import db_store
from db_core import engine

db_store.init_db()          # the server's startup step, done by hand here

client = TestClient(main.app)
NO_FOLLOW = {"follow_redirects": False}

# --- anonymous visitor -------------------------------------------------------
assert client.get("/health").status_code == 200

for page in ("/", "/capture"):
    r = client.get(page, **NO_FOLLOW)
    assert r.status_code == 303 and r.headers["location"] == "/login", page

assert client.post("/ingest", json={}).status_code == 401
tiny = {"session_name": "x",
        "accel": [{"t": 0, "x": 0, "y": 0, "z": 9.8}],
        "gyro": [{"t": 0, "x": 0, "y": 0, "z": 0}]}
assert client.post("/ingest", json=tiny).status_code == 401
print("anonymous visitors blocked: OK")

# --- after registering -------------------------------------------------------
r = client.post("/register", data={"username": "tester", "password": "password123",
                                   "confirm": "password123"}, **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/"
r = client.get("/")                                    # dashboard: the landing page
assert r.status_code == 200 and "Your progress" in r.text and "No sessions yet" in r.text
assert client.get("/capture").status_code == 200        # capture page, one tap away
assert client.get("/api/me").json()["username"] == "tester"
print("logged-in user reaches the dashboard and the capture page: OK")

print("\nAll main.py wiring checks passed.")
engine.dispose()