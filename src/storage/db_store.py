"""
src/storage/db_store.py

Purpose (Day 26, step 1b): every function the app calls to read or write the
database. It sits on top of db_core.py (engine + tables) and auth_utils.py
(hashing), and replaces the old db.py for the multi-user app.

What changed compared with the old db.py
----------------------------------------
* Every session belongs to a user: save_session() and list_sessions() take a
  user_id, and get_sets_for_session() refuses to return a session that belongs
  to someone else (prevents one user reading another's data by guessing an id).
* New account functions: create_user, authenticate.
* Session lookup (Day 26): get_session (used by the PDF report).
* Dashboard queries (Day 26): get_summary, get_recent_sessions,
  get_exercise_session_times (for the streak tiles).
* Weekly goal (Day 26): get_weekly_goal, set_weekly_goal, clear_weekly_goal.
* New login-token functions: create_login_token, get_user_by_token,
  delete_login_token.
* migrate_add_scoring_columns() is gone: the fresh database already has all
  columns.

Import style: the project adds src/storage to sys.path and imports modules by
plain name (same as the old `from db import ...`).
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import delete, exists, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from auth_utils import (
    hash_password, hash_token, new_token, normalize_username,
    validate_credentials, verify_password,
)
from db_core import engine, goals, init_db, login_tokens, sessions, sets, users

TOKEN_LIFETIME_DAYS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ================================================================== accounts ==
def create_user(username: str, password: str) -> int:
    """
    Register a new account and return its id.
    Raises ValueError with a user-friendly message if the username/password is
    invalid or the username is already taken.
    """
    username = normalize_username(username)
    error = validate_credentials(username, password)
    if error:
        raise ValueError(error)

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(users).values(
                    username=username,
                    password_hash=hash_password(password),
                    created_at=_now().isoformat(),
                )
            )
            return int(result.inserted_primary_key[0])
    except IntegrityError as e:
        raise ValueError("That username is already taken.") from e


_dummy_hash: str | None = None


def _get_dummy_hash() -> str:
    """A throw-away hash, built once, used to equalise login timing."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("not-a-real-password")
    return _dummy_hash


def authenticate(username: str, password: str) -> int | None:
    """Return the user's id if the credentials are correct, else None."""
    username = normalize_username(username)
    with engine.connect() as conn:
        row = conn.execute(
            select(users.c.id, users.c.password_hash)
            .where(users.c.username == username)
        ).first()

    if row is None:
        # Do the same slow hashing work as a real check, so the response time
        # does not reveal whether this username exists.
        verify_password(password, _get_dummy_hash())
        return None
    return int(row.id) if verify_password(password, row.password_hash) else None


# ============================================================ login tokens ==
def create_login_token(user_id: int) -> str:
    """
    Create a login for this user and return the RAW token (put it in the
    cookie). Only its hash is stored in the database.
    """
    token = new_token()
    now = _now()
    with engine.begin() as conn:
        conn.execute(
            insert(login_tokens).values(
                token_hash=hash_token(token),
                user_id=user_id,
                created_at=now.isoformat(),
                expires_at=(now + timedelta(days=TOKEN_LIFETIME_DAYS)).isoformat(),
            )
        )
    return token


def get_user_by_token(token: str | None) -> dict | None:
    """Return {'id', 'username'} for a valid, unexpired token; else None."""
    if not token:
        return None

    with engine.connect() as conn:
        row = conn.execute(
            select(users.c.id, users.c.username, login_tokens.c.expires_at)
            .join(login_tokens, login_tokens.c.user_id == users.c.id)
            .where(login_tokens.c.token_hash == hash_token(token))
        ).first()

    if row is None:
        return None
    if datetime.fromisoformat(row.expires_at) <= _now():
        delete_login_token(token)          # tidy up the expired row
        return None
    return {"id": int(row.id), "username": row.username}


def delete_login_token(token: str) -> None:
    """Log out: remove this login (other devices stay logged in)."""
    with engine.begin() as conn:
        conn.execute(
            delete(login_tokens).where(login_tokens.c.token_hash == hash_token(token))
        )


# ================================================== sessions and their sets ==
def _seconds_to_mmss(seconds: float) -> str:
    """37.5 -> '00:37'. Truncates sub-second precision (display only)."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _to_sql_real(value) -> float | None:
    """pandas NaN -> None (SQL NULL); NaN must never reach the database."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


_REQUIRED_COLS = {
    "label", "start_time", "end_time", "duration_s", "num_windows",
    "mean_confidence", "confidence_std", "raw_agreement", "is_short",
    "estimated_reps", "quality_score", "feedback",
}


def save_session(scored_df: pd.DataFrame, session_id: str, source_name: str,
                 user_id: int) -> None:
    """
    Persist one session's SCORED sets for `user_id` (all-or-nothing: if any
    insert fails, nothing is saved).

    scored_df is the output of quality_scorer.score_sets(), exactly as before.
    Raises ValueError for an empty/incorrect dataframe, a duplicate session_id,
    or an unknown user_id.
    """
    if scored_df.empty:
        raise ValueError("save_session received an empty scored_df — nothing to store.")

    missing = _REQUIRED_COLS - set(scored_df.columns)
    if missing:
        raise ValueError(
            f"save_session expects the SCORED dataframe (quality_scorer.score_sets "
            f"output). Missing columns: {sorted(missing)}."
        )

    session_start_epoch = float(scored_df["start_time"].iloc[0])
    session_end_epoch = float(scored_df["end_time"].iloc[-1])

    set_rows = []
    for idx, row in enumerate(scored_df.itertuples(index=False), start=1):
        elapsed_start_s = float(row.start_time) - session_start_epoch
        elapsed_end_s = float(row.end_time) - session_start_epoch
        set_rows.append({
            "session_id": session_id,
            "set_index": idx,
            "label": row.label,
            "start_time_epoch": float(row.start_time),
            "end_time_epoch": float(row.end_time),
            "elapsed_start_s": elapsed_start_s,
            "elapsed_end_s": elapsed_end_s,
            "start_mmss": _seconds_to_mmss(elapsed_start_s),
            "end_mmss": _seconds_to_mmss(elapsed_end_s),
            "duration_s": float(row.duration_s),
            "num_windows": int(row.num_windows),
            "mean_confidence": float(row.mean_confidence),
            "confidence_std": _to_sql_real(row.confidence_std),
            "raw_agreement": _to_sql_real(row.raw_agreement),
            "is_short": int(bool(row.is_short)),
            "estimated_reps": _to_sql_real(row.estimated_reps),
            "quality_score": _to_sql_real(row.quality_score),
            "feedback": None if pd.isna(row.feedback) else str(row.feedback),
        })

    try:
        with engine.begin() as conn:          # one transaction
            conn.execute(
                insert(sessions).values(
                    session_id=session_id,
                    user_id=user_id,
                    source_name=source_name,
                    uploaded_at=_now().isoformat(),
                    session_start_epoch=session_start_epoch,
                    total_duration_s=session_end_epoch - session_start_epoch,
                )
            )
            conn.execute(insert(sets), set_rows)
    except IntegrityError as e:
        raise ValueError(
            f"Could not save session '{session_id}': the id already exists "
            f"or the user does not exist."
        ) from e


def list_sessions(user_id: int) -> pd.DataFrame:
    """This user's sessions, most recent upload first."""
    query = (
        select(sessions)
        .where(sessions.c.user_id == user_id)
        .order_by(sessions.c.uploaded_at.desc())
    )
    return pd.read_sql_query(query, engine)


def get_session(session_id: str, user_id: int) -> dict | None:
    """
    One session's columns as a dict -- but only if it belongs to `user_id`.
    Returns None for an unknown id AND for someone else's session, so callers
    cannot tell the two apart (nothing leaks about other users' ids).
    """
    with engine.connect() as conn:
        row = conn.execute(
            select(sessions)
            .where(sessions.c.session_id == session_id, sessions.c.user_id == user_id)
        ).mappings().first()
    return None if row is None else dict(row)


def get_sets_for_session(session_id: str, user_id: int) -> pd.DataFrame:
    """
    All sets of one session in set_index order -- but only if the session
    belongs to `user_id`. For anyone else the result is simply empty.
    """
    query = (
        select(sets)
        .join(sessions, sets.c.session_id == sessions.c.session_id)
        .where(sets.c.session_id == session_id, sessions.c.user_id == user_id)
        .order_by(sets.c.set_index)
    )
    return pd.read_sql_query(query, engine)


# ================================================================ dashboard ==
# Rest periods carry no reps and no quality score, so dashboard numbers ignore them.
REST_LABEL = "non_activity"


def get_summary(user_id: int) -> dict:
    """
    Headline numbers for one user:
      sessions     - how many sessions they have saved
      total_reps   - estimated reps summed over all exercise sets (rounded)
      avg_quality  - mean quality score over all scored exercise sets,
                     or None if nothing has been scored yet
    """
    with engine.connect() as conn:
        n_sessions = conn.execute(
            select(func.count()).select_from(sessions)
            .where(sessions.c.user_id == user_id)
        ).scalar_one()

        total_reps, avg_quality = conn.execute(
            select(
                func.coalesce(func.sum(sets.c.estimated_reps), 0.0),
                func.avg(sets.c.quality_score),      # SQL AVG ignores NULLs
            )
            .select_from(sets.join(sessions, sets.c.session_id == sessions.c.session_id))
            .where(sessions.c.user_id == user_id, sets.c.label != REST_LABEL)
        ).one()

    return {
        "sessions": int(n_sessions),
        "total_reps": int(round(total_reps or 0)),
        "avg_quality": None if avg_quality is None else float(avg_quality),
    }


def get_recent_sessions(user_id: int, limit: int = 10) -> list[dict]:
    """
    The user's most recent sessions (newest first), each a dict of the session
    columns plus a "sets" list holding its EXERCISE sets in order (rest periods
    left out). Two queries in total, however many sessions are returned.
    """
    with engine.connect() as conn:
        session_rows = conn.execute(
            select(sessions)
            .where(sessions.c.user_id == user_id)
            .order_by(sessions.c.uploaded_at.desc())
            .limit(limit)
        ).mappings().all()

        session_ids = [row["session_id"] for row in session_rows]
        set_rows = []
        if session_ids:
            set_rows = conn.execute(
                select(sets)
                .where(sets.c.session_id.in_(session_ids), sets.c.label != REST_LABEL)
                .order_by(sets.c.session_id, sets.c.set_index)
            ).mappings().all()

    sets_by_session: dict[str, list[dict]] = {sid: [] for sid in session_ids}
    for row in set_rows:
        sets_by_session[row["session_id"]].append(dict(row))

    return [{**dict(row), "sets": sets_by_session[row["session_id"]]}
            for row in session_rows]


def get_exercise_session_times(user_id: int) -> list[str]:
    """
    Upload time (UTC ISO string, oldest first) of every session of this user
    that contains at least one real exercise set. Rest-only recordings are
    skipped, so they never count towards a streak. The caller converts the
    times to the user's local calendar days.
    """
    has_exercise_set = exists().where(
        sets.c.session_id == sessions.c.session_id,
        sets.c.label != REST_LABEL,
    )
    with engine.connect() as conn:
        rows = conn.execute(
            select(sessions.c.uploaded_at)
            .where(sessions.c.user_id == user_id, has_exercise_set)
            .order_by(sessions.c.uploaded_at)
        ).all()
    return [row[0] for row in rows]


# ==================================================================== goals ==
WEEKLY_SESSIONS_KIND = "sessions_per_week"
MAX_WEEKLY_GOAL = 21               # three sessions a day is already generous


def get_weekly_goal(user_id: int) -> int | None:
    """The user's target sessions per week, or None if they have not set one."""
    with engine.connect() as conn:
        row = conn.execute(
            select(goals.c.target)
            .where(goals.c.user_id == user_id, goals.c.kind == WEEKLY_SESSIONS_KIND)
        ).first()
    return None if row is None else int(row[0])


def set_weekly_goal(user_id: int, target: int) -> None:
    """
    Create or replace the user's weekly session goal.
    Raises ValueError unless target is a whole number from 1 to MAX_WEEKLY_GOAL,
    or if the user does not exist.
    """
    if (not isinstance(target, int) or isinstance(target, bool)
            or not 1 <= target <= MAX_WEEKLY_GOAL):
        raise ValueError(f"Goal must be a whole number from 1 to {MAX_WEEKLY_GOAL}.")

    now = _now().isoformat()
    try:
        with engine.begin() as conn:
            # "Upsert" written portably (works on SQLite and Postgres): try to
            # update the existing row first, insert only if there was none.
            updated = conn.execute(
                update(goals)
                .where(goals.c.user_id == user_id, goals.c.kind == WEEKLY_SESSIONS_KIND)
                .values(target=target, updated_at=now)
            ).rowcount
            if updated == 0:
                conn.execute(insert(goals).values(
                    user_id=user_id, kind=WEEKLY_SESSIONS_KIND,
                    target=target, updated_at=now))
    except IntegrityError as e:
        raise ValueError("Could not save the goal: the user does not exist.") from e


def clear_weekly_goal(user_id: int) -> None:
    """Remove the user's weekly goal (a no-op if there is none)."""
    with engine.begin() as conn:
        conn.execute(
            delete(goals)
            .where(goals.c.user_id == user_id, goals.c.kind == WEEKLY_SESSIONS_KIND)
        )