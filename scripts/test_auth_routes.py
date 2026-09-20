"""
scripts/test_auth_routes.py

Purpose: exercise the login flow end to end (register, login, logout, guards,
cookie flags) using FastAPI's TestClient -- no server, no browser, and a
THROWAWAY database, so your real rehab_planner_v2.db is untouched.

Run from the project root, inside your fastapi-env:
    python scripts/test_auth_routes.py

TestClient needs the `httpx` package (a dev-only need; do NOT add it to
backend/requirements.txt). If it complains about an unexpected keyword
argument 'app', run:  pip install "httpx<0.28"
"""

import os
import sys
import tempfile
from pathlib import Path

# --- throwaway database + clean cookie env, BEFORE importing the app --------
_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_tmp.name) / 'test.db').as_posix()}"
os.environ.pop("RENDER", None)
os.environ.pop("COOKIE_SECURE", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "storage"))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

import auth_routes
import db_store
from db_core import engine, login_tokens

db_store.init_db()

# A tiny app: the auth router plus one guarded page and one guarded API route.
app = FastAPI()
app.include_router(auth_routes.router)


@app.get("/secret")
def secret_page(user=Depends(auth_routes.require_user_page)):
    return {"page_for": user["username"]}


@app.get("/api/secret")
def secret_api(user=Depends(auth_routes.require_user_api)):
    return {"data_for": user["username"]}


client = TestClient(app)
NO_FOLLOW = {"follow_redirects": False}

# --- pages render -------------------------------------------------------------
assert client.get("/login").status_code == 200
assert "Create account" in client.get("/register").text

# --- guards block anonymous visitors --------------------------------------------
r = client.get("/secret", **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/login"
assert client.get("/api/secret").status_code == 401
assert client.get("/api/me").status_code == 401
print("pages + guards: OK")

# --- registration ----------------------------------------------------------------
r = client.post("/register", data={"username": "alice", "password": "password123",
                                   "confirm": "different"})
assert r.status_code == 400 and "do not match" in r.text

r = client.post("/register", data={"username": "alice", "password": "short",
                                   "confirm": "short"})
assert r.status_code == 400 and "at least" in r.text

# HTML in a username must come back escaped, never as live markup (XSS).
r = client.post("/register", data={"username": "<script>alert(1)</script>",
                                   "password": "password123",
                                   "confirm": "password123"})
assert r.status_code == 400
assert "<script>alert(1)</script>" not in r.text and "&lt;script&gt;" in r.text

r = client.post("/register", data={"username": "alice", "password": "password123",
                                   "confirm": "password123"}, **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/"
cookie_header = r.headers["set-cookie"].lower()
assert "httponly" in cookie_header and "samesite=lax" in cookie_header
assert "secure" not in cookie_header.replace("samesite", "")   # local http: not Secure
print("register: OK")

# --- logged in ---------------------------------------------------------------------
assert client.get("/secret").json() == {"page_for": "alice"}
assert client.get("/api/secret").json() == {"data_for": "alice"}
assert client.get("/api/me").json()["username"] == "alice"
assert client.get("/login", **NO_FOLLOW).status_code == 303   # already logged in

# The raw cookie value must never be stored in the database (only its hash).
raw_token = client.cookies.get(auth_routes.COOKIE_NAME)
with engine.connect() as conn:
    stored = [row[0] for row in conn.execute(select(login_tokens.c.token_hash))]
assert raw_token and raw_token not in stored and len(stored) == 1

# --- duplicate username ---------------------------------------------------------------
other = TestClient(app)
r = other.post("/register", data={"username": "ALICE", "password": "password123",
                                  "confirm": "password123"})
assert r.status_code == 400 and "already taken" in r.text
print("session/cookie: OK")

# --- logout ----------------------------------------------------------------------------
r = client.post("/logout", **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/login"
assert client.get("/secret", **NO_FOLLOW).status_code == 303
assert client.get("/api/me").status_code == 401
print("logout: OK")

# --- login -------------------------------------------------------------------------------
r = client.post("/login", data={"username": "alice", "password": "wrong-password"})
assert r.status_code == 401 and "Wrong username or password." in r.text
r = client.post("/login", data={"username": "nobody", "password": "password123"})
assert r.status_code == 401 and "Wrong username or password." in r.text   # same message

r = client.post("/login", data={"username": " Alice ", "password": "password123"},
                **NO_FOLLOW)
assert r.status_code == 303
assert client.get("/api/me").json()["username"] == "alice"
print("login: OK")

# --- Secure flag switches on for Render (HTTPS) -------------------------------------------
os.environ["RENDER"] = "true"
r = TestClient(app).post("/login", data={"username": "alice", "password": "password123"},
                         **NO_FOLLOW)
assert "; secure" in r.headers["set-cookie"].lower()
os.environ.pop("RENDER")
print("secure cookie on Render: OK")

print("\nAll auth_routes checks passed.")
engine.dispose()
