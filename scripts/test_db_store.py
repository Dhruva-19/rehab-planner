"""
scripts/test_db_store.py

Purpose: verify db_store.py end to end. It runs against a THROWAWAY database
in a temp folder, so your real rehab_planner_v2.db is never touched.

Run from the project root:   py -3.12 scripts/test_db_store.py
"""

import os
import sys
import tempfile
from pathlib import Path

# --- point the app at a throwaway database BEFORE importing db_core ---------
_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_tmp.name) / 'test.db').as_posix()}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "storage"))

import pandas as pd
from sqlalchemy import update

import db_store as store
from db_core import engine, login_tokens


def expect_error(fn, *args, **kwargs):
    """Assert that fn raises ValueError; return the message."""
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} should have raised ValueError")


def make_scored_df() -> pd.DataFrame:
    """A tiny SCORED dataframe with two sets (one with NaNs, like a rest set)."""
    return pd.DataFrame({
        "label": ["squats", "non_activity"],
        "start_time": [1_700_000_000.0, 1_700_000_030.0],
        "end_time": [1_700_000_030.0, 1_700_000_045.0],
        "duration_s": [30.0, 15.0],
        "num_windows": [10, 5],
        "mean_confidence": [0.9, 0.8],
        "confidence_std": [0.05, float("nan")],
        "raw_agreement": [0.95, float("nan")],
        "is_short": [False, False],
        "estimated_reps": [9.0, float("nan")],
        "quality_score": [95.0, float("nan")],
        "feedback": ["Good form and consistency.", None],
    })


store.init_db()

# --- accounts ---------------------------------------------------------------
a = store.create_user("  Alice ", "password123")          # normalised to 'alice'
b = store.create_user("bob_1", "password456")
assert a != b
assert "already taken" in expect_error(store.create_user, "ALICE", "another-pass-1")
assert expect_error(store.create_user, "x!", "password123")     # bad username
assert expect_error(store.create_user, "carol", "short")        # bad password

assert store.authenticate("alice", "password123") == a
assert store.authenticate("ALICE ", "password123") == a         # normalised
assert store.authenticate("alice", "wrong-password") is None
assert store.authenticate("nobody", "password123") is None
print("accounts: OK")

# --- login tokens -----------------------------------------------------------
token = store.create_login_token(a)
assert store.get_user_by_token(token) == {"id": a, "username": "alice"}
assert store.get_user_by_token("garbage") is None
assert store.get_user_by_token(None) is None

store.delete_login_token(token)                                 # logout
assert store.get_user_by_token(token) is None

token2 = store.create_login_token(a)                            # expired token
with engine.begin() as conn:
    conn.execute(update(login_tokens).values(expires_at="2000-01-01T00:00:00+00:00"))
assert store.get_user_by_token(token2) is None
print("login tokens: OK")

# --- sessions and sets (per-user scoping) ------------------------------------
store.save_session(make_scored_df(), "s1", "Squats.csv", user_id=a)

assert len(store.list_sessions(a)) == 1
assert len(store.list_sessions(b)) == 0                          # bob sees nothing

mine = store.get_sets_for_session("s1", a)
assert list(mine["label"]) == ["squats", "non_activity"]
assert mine["start_mmss"].tolist() == ["00:00", "00:30"]
assert pd.isna(mine.loc[1, "quality_score"])                     # NaN stored as NULL
assert store.get_sets_for_session("s1", b).empty                 # ownership check

one = store.get_session("s1", a)                                 # single-session lookup
assert one["session_id"] == "s1" and one["source_name"] == "Squats.csv" and one["user_id"] == a
assert store.get_session("s1", b) is None                        # someone else's -> None
assert store.get_session("nope", a) is None                      # unknown -> None (same answer)

assert "id already exists" in expect_error(
    store.save_session, make_scored_df(), "s1", "dup.csv", a)
assert "user does not exist" in expect_error(
    store.save_session, make_scored_df(), "s2", "x.csv", 999)
assert len(store.list_sessions(a)) == 1                          # failed saves left nothing
print("sessions/sets: OK")

# --- dashboard queries ---------------------------------------------------------
summary = store.get_summary(a)
assert summary == {"sessions": 1, "total_reps": 9, "avg_quality": 95.0}, summary
assert store.get_summary(b) == {"sessions": 0, "total_reps": 0, "avg_quality": None}

recent = store.get_recent_sessions(a)
assert len(recent) == 1 and recent[0]["session_id"] == "s1"
assert [x["label"] for x in recent[0]["sets"]] == ["squats"]      # rest set left out
assert store.get_recent_sessions(b) == []                          # bob sees nothing

second = make_scored_df()
second.loc[0, "quality_score"] = 85.0                              # 2nd session, lower score
second.loc[0, "estimated_reps"] = 11.0
store.save_session(second, "s2", "Squats2.csv", user_id=a)
recent = store.get_recent_sessions(a)
assert [x["session_id"] for x in recent] == ["s2", "s1"]           # newest first
assert len(store.get_recent_sessions(a, limit=1)) == 1
summary = store.get_summary(a)
assert summary["sessions"] == 2 and summary["total_reps"] == 20
assert summary["avg_quality"] == 90.0                              # mean of 95 and 85
print("dashboard queries: OK")

# --- exercise-session times (for streaks) -----------------------------------------
rest_only = make_scored_df().iloc[[1]].reset_index(drop=True)      # only the non_activity row
store.save_session(rest_only, "s3", "Rest.csv", user_id=a)
times = store.get_exercise_session_times(a)
assert len(times) == 2                                              # s1 and s2; rest-only s3 skipped
assert times == sorted(times)                                       # oldest first
assert store.get_exercise_session_times(b) == []
print("exercise session times: OK")

# --- weekly goal ---------------------------------------------------------------------
assert store.get_weekly_goal(a) is None
store.set_weekly_goal(a, 3)
assert store.get_weekly_goal(a) == 3
store.set_weekly_goal(a, 5)                                         # replaces, no duplicate row
assert store.get_weekly_goal(a) == 5
assert store.get_weekly_goal(b) is None                             # per-user
for bad in (0, -1, store.MAX_WEEKLY_GOAL + 1, "3", 2.5, True, None):
    assert "whole number" in expect_error(store.set_weekly_goal, a, bad), bad
assert store.get_weekly_goal(a) == 5                                # bad input changed nothing
assert "does not exist" in expect_error(store.set_weekly_goal, 999, 3)
store.set_weekly_goal(b, 1)
store.clear_weekly_goal(a)
assert store.get_weekly_goal(a) is None and store.get_weekly_goal(b) == 1
store.clear_weekly_goal(a)                                          # clearing twice is fine
print("weekly goal: OK")

# --- deployment safety: on Render a missing DATABASE_URL must fail loudly ----------------
import db_core

saved = {k: os.environ.get(k) for k in ("DATABASE_URL", "RENDER")}
try:
    os.environ.pop("DATABASE_URL", None)
    os.environ["RENDER"] = "true"
    try:
        db_core._get_database_url()
        raise AssertionError("expected a RuntimeError on Render without DATABASE_URL")
    except RuntimeError as e:
        assert "DATABASE_URL is not set" in str(e)

    os.environ["DATABASE_URL"] = "postgres://user:pw@host/db?sslmode=require"      # Render + URL: fine
    assert db_core._get_database_url() == "postgresql+psycopg2://user:pw@host/db?sslmode=require"
finally:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
print("Render without DATABASE_URL fails loudly: OK")

print("\nAll db_store checks passed.")
engine.dispose()