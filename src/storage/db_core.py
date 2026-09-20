"""
src/storage/db_core.py

Purpose (Day 26, step 1a): the database foundation for the multi-user app.
This file only defines WHERE the database lives and WHAT tables it has.
It does not contain any save/load logic -- that is ported from the old
db.py in step 1b.

Design decisions
----------------
* One code path for two databases. The connection string comes from the
  DATABASE_URL environment variable:
    - not set  -> local SQLite file  database/rehab_planner_v2.db  (your PC)
    - set      -> hosted Postgres (e.g. Neon), used on Render
  The old database/rehab_planner.db is deliberately left untouched as a backup.
* Tables are declared with SQLAlchemy Core, so SQLAlchemy generates the right
  SQL for whichever database is in use (AUTOINCREMENT vs SERIAL, etc.).
* `sessions.user_id` is NOT NULL: every workout session must belong to a
  user. This closes the old "/ingest doesn't tag user_id" gap by design --
  saving without a user will fail loudly instead of silently.
* The file is named db_core.py (not database.py) so it can never be confused
  with the `database/` data folder in the project.
"""

import os
from pathlib import Path

from sqlalchemy import (
    Column, Float, ForeignKey, Index, Integer, MetaData, Table, Text,
    create_engine, event, inspect,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "database" / "rehab_planner_v2.db"


def _get_database_url() -> str:
    """Pick the connection string: DATABASE_URL if set, else local SQLite."""
    url = os.environ.get("DATABASE_URL", "").strip()

    if not url:
        DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # as_posix() keeps Windows paths in the "C:/Users/..." form SQLAlchemy expects.
        return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

    # Hosted providers usually hand out "postgres://" or "postgresql://".
    # SQLAlchemy needs to be told which Python driver to use (psycopg2).
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = _get_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    # pool_pre_ping: test a connection before using it, so a hosted database
    # that went idle and dropped the connection is reconnected automatically.
    pool_pre_ping=True,
    # FastAPI handles requests on several threads; SQLite must allow that.
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
)

if IS_SQLITE:
    # SQLite ignores FOREIGN KEY constraints unless this is switched on for
    # EVERY new connection (the old db.py did this in get_connection()).
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


metadata = MetaData()

# --------------------------------------------------------------------------
# users -- one row per registered account
# --------------------------------------------------------------------------
# username is stored lower-case (see auth_utils.normalize_username) so "Dhruva"
# and "dhruva" are the same account.
# password_hash is ONE self-describing string:
#   pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
# so no separate salt column is needed and the iteration count can be raised
# later without breaking existing accounts.
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),          # ISO-8601 UTC string
)

# --------------------------------------------------------------------------
# login_tokens -- one row per active login (one per device/browser)
# --------------------------------------------------------------------------
# Named login_tokens (not "sessions") because `sessions` already means a
# workout upload in this project.
# We store only the SHA-256 HASH of the token. The raw token lives only in the
# user's cookie, so a leaked database cannot be used to hijack logins.
login_tokens = Table(
    "login_tokens", metadata,
    Column("token_hash", Text, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("created_at", Text, nullable=False),
    Column("expires_at", Text, nullable=False),
)
Index("idx_login_tokens_user", login_tokens.c.user_id)

# --------------------------------------------------------------------------
# sessions -- one row per upload / live capture (same columns as before,
# plus user_id)
# --------------------------------------------------------------------------
sessions = Table(
    "sessions", metadata,
    Column("session_id", Text, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("source_name", Text, nullable=False),
    Column("uploaded_at", Text, nullable=False),
    Column("session_start_epoch", Float, nullable=False),
    Column("total_duration_s", Float, nullable=False),
)
Index("idx_sessions_user", sessions.c.user_id)

# --------------------------------------------------------------------------
# sets -- one row per detected exercise set (unchanged from the old schema)
# --------------------------------------------------------------------------
sets = Table(
    "sets", metadata,
    Column("set_id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", Text, ForeignKey("sessions.session_id"), nullable=False),
    Column("set_index", Integer, nullable=False),
    Column("label", Text, nullable=False),
    Column("start_time_epoch", Float, nullable=False),
    Column("end_time_epoch", Float, nullable=False),
    Column("elapsed_start_s", Float, nullable=False),
    Column("elapsed_end_s", Float, nullable=False),
    Column("start_mmss", Text, nullable=False),
    Column("end_mmss", Text, nullable=False),
    Column("duration_s", Float, nullable=False),
    Column("num_windows", Integer, nullable=False),
    Column("mean_confidence", Float, nullable=False),
    Column("confidence_std", Float),          # nullable: NaN on single-window sets
    Column("raw_agreement", Float),           # nullable
    Column("is_short", Integer, nullable=False),
    Column("estimated_reps", Float),          # nullable
    Column("quality_score", Float),           # nullable: not scored for rest periods
    Column("feedback", Text),                 # nullable
)
Index("idx_sets_session", sets.c.session_id)

# --------------------------------------------------------------------------
# goals -- one target per user per goal type
# --------------------------------------------------------------------------
# Version one has a single kind, "sessions_per_week". Keying by (user, kind)
# means per-exercise goals (e.g. "squats_reps_per_week") can be added later
# without changing the table. init_db() creates this table on an existing
# database automatically; no migration is needed.
goals = Table(
    "goals", metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("kind", Text, primary_key=True),
    Column("target", Integer, nullable=False),
    Column("updated_at", Text, nullable=False),          # ISO-8601 UTC string
)


def init_db() -> None:
    """Create any missing tables. Safe to call on every startup."""
    metadata.create_all(engine)


if __name__ == "__main__":
    # Smoke test: create the tables and print what exists.
    init_db()
    print(f"Database URL : {engine.url.render_as_string(hide_password=True)}")
    insp = inspect(engine)
    for table_name in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(table_name)]
        print(f"  {table_name}: {cols}")