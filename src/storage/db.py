"""
src/storage/db.py

Purpose: Persist SCORED exercise "sets" (output of quality_scorer.score_sets,
which itself runs on aggregate_sets.aggregate_into_sets output) into SQLite,
so a Streamlit dashboard can query past sessions -- including their quality
scores and feedback -- without re-running inference.

Design: two tables -- sessions (one row per upload) and sets (many rows per
session, one per exercise set). Elapsed/mm:ss display columns are computed
once at write time so the dashboard layer never has to touch epoch-time math.

CHANGE FROM DAY 7 VERSION: the `sets` table now also stores confidence_std,
raw_agreement, quality_score, and feedback -- the Day 8 quality_scorer
output. Without this, a session reloaded from the DB (dashboard "Past
Sessions" tab) would have no way to show quality scores, because the
raw per-window predictions needed to recompute confidence_std/raw_agreement
aren't stored anywhere. So save_session() now expects the SCORED dataframe
(output of quality_scorer.score_sets), not the raw aggregated one.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "database" / "rehab_planner.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    source_name         TEXT NOT NULL,
    uploaded_at          TEXT NOT NULL,
    session_start_epoch  REAL NOT NULL,
    total_duration_s     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sets (
    set_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    set_index         INTEGER NOT NULL,
    label             TEXT NOT NULL,
    start_time_epoch  REAL NOT NULL,
    end_time_epoch    REAL NOT NULL,
    elapsed_start_s   REAL NOT NULL,
    elapsed_end_s     REAL NOT NULL,
    start_mmss        TEXT NOT NULL,
    end_mmss          TEXT NOT NULL,
    duration_s        REAL NOT NULL,
    num_windows       INTEGER NOT NULL,
    mean_confidence   REAL NOT NULL,
    confidence_std    REAL,
    raw_agreement     REAL,
    is_short          INTEGER NOT NULL,
    estimated_reps    REAL,
    quality_score     REAL,
    feedback          TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);

CREATE INDEX IF NOT EXISTS idx_sets_session ON sets (session_id);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign key enforcement turned on."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't already exist. Safe to call every run."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def migrate_add_scoring_columns(db_path: str = DB_PATH) -> None:
    """
    One-time migration for DBs created before Day 9 (i.e. before
    confidence_std/raw_agreement/quality_score/feedback existed).
    Safe to call repeatedly -- skips columns that already exist.

    SQLite has no "ADD COLUMN IF NOT EXISTS", so we check the existing
    schema first via PRAGMA table_info.
    """
    conn = get_connection(db_path)
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sets)")}
        new_cols = {
            "confidence_std": "REAL",
            "raw_agreement": "REAL",
            "quality_score": "REAL",
            "feedback": "TEXT",
            "estimated_reps": "REAL",
        }
        for col, col_type in new_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE sets ADD COLUMN {col} {col_type}")
        conn.commit()
    finally:
        conn.close()


def _seconds_to_mmss(seconds: float) -> str:
    """37.5 -> '00:37'. Truncates sub-second precision (display only)."""
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _to_sql_real(value) -> float | None:
    """
    pandas NaN -> SQL NULL. Needed because quality_score/confidence_std/
    raw_agreement can legitimately be NaN (e.g. non_activity sets, or
    confidence_std on single-window sets) -- sqlite3 stores Python None
    as NULL, but a float NaN gets stored as the string/float NaN itself,
    which breaks later reads/comparisons. Always route REAL-but-nullable
    columns through this before binding.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return float(value)


def save_session(scored_df: pd.DataFrame,
                  session_id: str,
                  source_name: str,
                  db_path: str = DB_PATH) -> None:
    """
    Persist one session's SCORED sets to the database.

    Parameters
    ----------
    scored_df : output of quality_scorer.score_sets() (which itself takes
        aggregate_sets.aggregate_into_sets() output). Must be non-empty
        and sorted by start_time (aggregate_into_sets already guarantees
        this; score_sets preserves row order).
    session_id : caller-supplied unique id for this upload, e.g.
        f"{source_name}_{datetime.now():%Y%m%d_%H%M%S}".
    source_name : human-readable label, e.g. the uploaded filename.
    """
    if scored_df.empty:
        raise ValueError("save_session received an empty scored_df — nothing to store.")

    required_cols = {
        "label", "start_time", "end_time", "duration_s", "num_windows",
        "mean_confidence", "confidence_std", "raw_agreement", "is_short",
        "estimated_reps", "quality_score", "feedback",
    }
    missing = required_cols - set(scored_df.columns)
    if missing:
        raise ValueError(
            f"save_session expects the SCORED dataframe (quality_scorer.score_sets "
            f"output). Missing columns: {sorted(missing)}. Did you pass the raw "
            f"aggregate_into_sets() output instead?"
        )

    conn = get_connection(db_path)
    try:
        session_start_epoch = float(scored_df["start_time"].iloc[0])
        session_end_epoch = float(scored_df["end_time"].iloc[-1])
        total_duration_s = session_end_epoch - session_start_epoch

        conn.execute(
            """INSERT INTO sessions
               (session_id, source_name, uploaded_at, session_start_epoch, total_duration_s)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session_id,
                source_name,
                datetime.now(timezone.utc).isoformat(),
                session_start_epoch,
                total_duration_s,
            ),
        )

        rows = []
        for idx, row in enumerate(scored_df.itertuples(index=False), start=1):
            elapsed_start_s = row.start_time - session_start_epoch
            elapsed_end_s = row.end_time - session_start_epoch
            rows.append((
                session_id,
                idx,
                row.label,
                float(row.start_time),
                float(row.end_time),
                elapsed_start_s,
                elapsed_end_s,
                _seconds_to_mmss(elapsed_start_s),
                _seconds_to_mmss(elapsed_end_s),
                float(row.duration_s),
                int(row.num_windows),
                float(row.mean_confidence),
                _to_sql_real(row.confidence_std),
                _to_sql_real(row.raw_agreement),
                int(bool(row.is_short)),
                _to_sql_real(row.estimated_reps),
                _to_sql_real(row.quality_score),
                None if pd.isna(row.feedback) else str(row.feedback),
            ))

        conn.executemany(
            """INSERT INTO sets
               (session_id, set_index, label, start_time_epoch, end_time_epoch,
                elapsed_start_s, elapsed_end_s, start_mmss, end_mmss,
                duration_s, num_windows, mean_confidence, confidence_std,
                raw_agreement, is_short, estimated_reps, quality_score, feedback)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise ValueError(
            f"session_id '{session_id}' already exists — choose a unique id "
            f"(e.g. include a timestamp)."
        ) from e
    finally:
        conn.close()


def list_sessions(db_path: str = DB_PATH) -> pd.DataFrame:
    """All sessions, most recent upload first."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM sessions ORDER BY uploaded_at DESC", conn
        )
    finally:
        conn.close()


def get_sets_for_session(session_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """All sets for one session, in original set_index order."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM sets WHERE session_id = ? ORDER BY set_index",
            conn,
            params=(session_id,),
        )
    finally:
        conn.close()


if __name__ == "__main__":
    # Manual smoke test -- wires Day 6 pipeline -> Day 7 aggregation ->
    # Day 8 scoring -> DB.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "inference"))
    sys.path.append(str(Path(__file__).resolve().parents[1] / "feedback"))
    from predict_pipeline import predict_from_raw_csv
    from aggregate_sets import aggregate_into_sets
    from quality_scorer import score_sets, session_summary

    ACC_CSV = "data/sample_upload/session_acc.csv"
    GYRO_CSV = "data/sample_upload/session_gyro.csv"

    init_db()
    migrate_add_scoring_columns()  # no-op on a fresh DB, fixes an old one

    window_results = predict_from_raw_csv(ACC_CSV, GYRO_CSV)
    session_sets = aggregate_into_sets(window_results)
    scored = score_sets(session_sets)

    sid = f"w05_test_{datetime.now():%Y%m%d_%H%M%S}"
    save_session(scored, session_id=sid, source_name="w05 sample upload")

    print(f"\nSaved session '{sid}' with {len(scored)} sets.\n")
    print("All sessions in DB:")
    print(list_sessions()[["session_id", "source_name", "uploaded_at", "total_duration_s"]])

    print(f"\nSets for '{sid}':")
    reloaded = get_sets_for_session(sid)
    print(reloaded[["set_index", "label", "start_mmss", "end_mmss",
                     "duration_s", "mean_confidence", "is_short",
                     "quality_score", "feedback"]])
    print("\nSession summary (recomputed from reloaded DB rows):")
    print(session_summary(reloaded))