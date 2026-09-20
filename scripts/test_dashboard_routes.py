"""
scripts/test_dashboard_routes.py

Purpose: check the dashboard page end to end (login guard, empty state, real
numbers, streak tiles and their local-time-zone handling, the weekly goal card
and Goals page, per-user isolation, HTML escaping, the "latest 10" limit) using
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

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import update

import auth_routes
import dashboard_routes
import db_store
from db_core import engine, sessions as sessions_table

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
assert anon.get("/goals", **NO_FOLLOW).status_code == 303
r = anon.post("/goals", data={"target": "3"}, **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/login"
assert anon.post("/goals/clear", **NO_FOLLOW).status_code == 303
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
assert '<span class="num">0</span><span class="lbl">Day streak</span>' in r.text
assert "Record a session today to start a streak." in r.text
assert "Set a weekly goal" in r.text and 'href="/goals"' in r.text        # no goal yet
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
assert '<span class="num">1</span><span class="lbl">Day streak</span>' in page      # both saved today
assert '<span class="num">1</span><span class="lbl">Best streak</span>' in page
assert '<span class="num">2</span><span class="lbl">This week</span>' in page
assert "You have trained today." in page
assert "tz_offset_min" in page and dashboard_routes.TZ_COOKIE == "tz_offset_min"
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

# --- streaks count LOCAL days (time-zone cookie), with a frozen clock ----------------------
carol = TestClient(app)
register(carol, "carol")
carol_id = carol.get("/api/me").json()["id"]
upload_times = ["2026-09-19T20:00:00+00:00",     # 01:30 on the 20th in India
                "2026-09-20T18:00:00+00:00",     # 23:30 on the 20th in India
                "2026-09-21T02:00:00+00:00"]     # 07:30 on the 21st in India
for i, when in enumerate(upload_times):
    db_store.save_session(scored_df(), f"c{i}", f"C{i}", carol_id)
    with engine.begin() as conn:
        conn.execute(update(sessions_table)
                     .where(sessions_table.c.session_id == f"c{i}")
                     .values(uploaded_at=when))


def carol_page(now_utc, offset):
    """Render carol's dashboard at a chosen 'now', with an optional tz cookie."""
    dashboard_routes._utc_now = lambda: now_utc
    token = carol.cookies.get(auth_routes.COOKIE_NAME)
    cookie = f"{auth_routes.COOKIE_NAME}={token}"
    if offset is not None:
        cookie += f"; tz_offset_min={offset}"
    return carol.get("/", headers={"Cookie": cookie}).text


def tile(page, value, label):
    return f'<span class="num">{value}</span><span class="lbl">{label}</span>' in page


monday_morning = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc)

page = carol_page(monday_morning, None)                  # no cookie -> UTC days 19, 20, 21
assert tile(page, 3, "Day streak") and tile(page, 3, "Best streak") and tile(page, 1, "This week")
assert carol_page(monday_morning, "abc") == page         # garbage cookie -> falls back to UTC

page = carol_page(monday_morning, 330)                   # India -> days 20, 20, 21
assert tile(page, 2, "Day streak") and tile(page, 2, "Best streak") and tile(page, 1, "This week")
assert "You have trained today." in page

page = carol_page(datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc), 0)   # nothing yet today
assert tile(page, 3, "Day streak") and "Exercise today to keep your 3-day streak alive." in page

page = carol_page(datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc), 0)   # two days missed
assert tile(page, 0, "Day streak") and tile(page, 3, "Best streak")
assert "Record a session today to start a streak." in page
print("streaks + time zones: OK")

# --- weekly goal: Goals page + progress card (frozen clock: Monday 21 Sept, India) ----------
now = monday_morning                                    # carol has 1 session this week (India days)
page = carol_page(now, 330)
assert "Set a weekly goal" in page and 'class="fill"' not in page

r = carol.get("/goals")
assert r.status_code == 200 and "Sessions per week" in r.text
assert 'action="/goals"' in r.text and 'class="pill"' in r.text
assert "Remove goal" not in r.text                                       # nothing to remove yet
assert r.headers["cache-control"] == "no-store"

r = carol.post("/goals", data={"target": "3"}, **NO_FOLLOW)              # save a goal of 3
assert r.status_code == 303 and r.headers["location"] == "/"
page = carol_page(now, 330)
assert '<b>1</b> / 3 sessions' in page
assert 'style="width:33%"' in page and 'aria-valuenow="33"' in page
assert "2 more to go" in page and "7 days left this week" in page        # Monday: 7 days incl. today
assert 'value="3"' in carol.get("/goals").text and "Remove goal" in carol.get("/goals").text

page = carol_page(datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc), 330)   # Sunday: 1 day left
assert "1 day left this week" in page and "1 days" not in page

carol.post("/goals", data={"target": "1"}, **NO_FOLLOW)                  # change to 1 -> reached
page = carol_page(now, 330)
assert "Goal reached. Great work!" in page and 'style="width:100%"' in page

for bad in ("0", "22", "-3", "abc", "2.5", ""):
    r = carol.post("/goals", data={"target": bad}, **NO_FOLLOW)
    assert r.status_code == 400 and "whole number from 1 to 21" in r.text, bad
assert '<b>1</b> / 1 sessions' in carol_page(now, 330)                   # bad input changed nothing

bob_page = bob_client.get("/").text                                       # goals are per user
assert "Set a weekly goal" in bob_page and "Goal reached" not in bob_page

r = carol.post("/goals/clear", **NO_FOLLOW)
assert r.status_code == 303 and r.headers["location"] == "/"
assert "Set a weekly goal" in carol_page(now, 330)
print("weekly goal: OK")

print("\nAll dashboard checks passed.")
engine.dispose()