"""
app.py

AI Personalized Rehabilitation Planner -- Streamlit Dashboard (Day 9)

Two tabs:
  1. New Session   : upload accelerometer + gyroscope CSVs, run the full
                      pipeline (predict -> aggregate into sets -> score
                      quality), save to DB, display results.
  2. Past Sessions  : pick a previously-saved session from the DB and view
                      its results again, no re-computation needed.

Run with:
    streamlit run app.py
from the project root (so the "database/rehab_planner.db" relative path
in db.py resolves correctly).
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# --- Make sibling src/ packages importable -----------------------------
# Mirrors the sys.path pattern already used in the __main__ blocks of
# predict_pipeline.py / aggregate_sets.py / quality_scorer.py / db.py.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src" / "inference"))
sys.path.append(str(PROJECT_ROOT / "src" / "feedback"))
sys.path.append(str(PROJECT_ROOT / "src" / "storage"))

from predict_pipeline import predict_from_raw_csv          # noqa: E402
from aggregate_sets import aggregate_into_sets              # noqa: E402
from quality_scorer import score_sets, session_summary      # noqa: E402
from db import (                                            # noqa: E402
    init_db,
    migrate_add_scoring_columns,
    save_session,
    list_sessions,
    get_sets_for_session,
)

# --- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="AI Rehab Planner",
    page_icon="🏋️",
    layout="wide",
)

# Make sure the DB exists and is on the latest schema every time the app boots.
init_db()
migrate_add_scoring_columns()


# =========================================================================
# Shared display helpers
# =========================================================================

DISPLAY_COLUMNS = [
    "set_index", "label", "start_mmss", "end_mmss", "duration_s",
    "num_windows", "mean_confidence", "is_short", "estimated_reps",
    "quality_score", "feedback",
]

COLUMN_LABELS = {
    "set_index": "Set #",
    "label": "Exercise",
    "start_mmss": "Start",
    "end_mmss": "End",
    "duration_s": "Duration (s)",
    "num_windows": "Windows",
    "mean_confidence": "Confidence",
    "is_short": "Short set?",
    "estimated_reps": "Reps (est.)",
    "quality_score": "Quality Score",
    "feedback": "Feedback",
}


def _score_color(score):
    """
    Map a quality_score (0-100 or NaN) to a background color.
    Bands mirror quality_scorer.FEEDBACK_BANDS (85 / 60 / 0).
    NaN (non_activity, unscored) gets a neutral grey.
    """
    if pd.isna(score):
        return "background-color: #e0e0e0"  # grey -- not scored
    if score >= 85:
        return "background-color: #b7e4c7"  # green -- good form
    if score >= 60:
        return "background-color: #ffe8a3"  # yellow -- some inconsistency
    return "background-color: #ffb3b3"      # red -- unstable, review form


def render_results(scored_df: pd.DataFrame, summary: dict, session_label: str):
    """
    Render the session summary + color-coded per-set table.
    Shared by both tabs so New Session and Past Sessions look identical.
    """
    st.subheader(f"Results: {session_label}")

    # --- Session summary cards -------------------------------------
    col1, col2, col3 = st.columns(3)
    avg_score = summary["avg_quality_score"]
    col1.metric(
        "Average Quality Score",
        f"{avg_score:.1f}" if avg_score is not None else "N/A",
    )
    col2.metric("Scored Sets", summary["num_scored_sets"])
    col3.metric("Short Sets Flagged", summary["num_short_sets"])

    st.markdown("---")

    # --- Per-set table, color-coded by quality_score ----------------
    display_df = scored_df[DISPLAY_COLUMNS].copy()
    display_df["is_short"] = display_df["is_short"].map({1: "Yes", 0: "No", True: "Yes", False: "No"})
    display_df["mean_confidence"] = display_df["mean_confidence"].round(3)
    display_df["duration_s"] = display_df["duration_s"].round(1)
    display_df["quality_score"] = pd.to_numeric(display_df["quality_score"], errors="coerce").round(1)

    # Convert Quality Score to a DISPLAY STRING ourselves before styling.
    # st.dataframe() does not reliably respect a Styler's .format(na_rep=...)
    # for missing values -- it shows the raw NaN as the literal text "None"
    # regardless of the format string. Pre-formatting sidesteps that.
    raw_scores = display_df["quality_score"]  # keep a numeric copy for coloring
    display_df["quality_score"] = raw_scores.apply(
        lambda v: "—" if pd.isna(v) else f"{v:.1f}"
    )

    display_df = display_df.rename(columns=COLUMN_LABELS)

    def _score_color_from_display(val):
        """Same color bands as _score_color, but reads the string we just built."""
        if val == "—":
            return "background-color: #e0e0e0"
        score = float(val)
        if score >= 85:
            return "background-color: #b7e4c7"
        if score >= 60:
            return "background-color: #ffe8a3"
        return "background-color: #ffb3b3"

    styled = display_df.style.map(_score_color_from_display, subset=["Quality Score"])
    styled = styled.format({
        "Duration (s)": "{:.1f}",
        "Confidence": "{:.3f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        "🟩 Good form (score ≥ 85)   🟨 Some inconsistency (60–84)   "
        "🟥 Unstable, review form (< 60)   ⬜ Not scored (rest period)"
    )

    print("DEBUG quality_score dtype:", display_df["Quality Score" if "Quality Score" in display_df.columns else "quality_score"].dtype)
# =========================================================================
# Tab 1: New Session
# =========================================================================

def new_session_tab():
    st.header("Upload a New Session")
    st.write(
        "Upload the accelerometer and gyroscope CSVs for one recording "
        "session (smartphone `sp_r` stream, MM-Fit format)."
    )

    col1, col2 = st.columns(2)
    acc_file = col1.file_uploader("Accelerometer CSV", type="csv", key="acc_upload")
    gyro_file = col2.file_uploader("Gyroscope CSV", type="csv", key="gyro_upload")

    source_name = st.text_input(
        "Session name (for your reference)",
        value=acc_file.name.rsplit(".", 1)[0] if acc_file else "",
        placeholder="e.g. patient1_session3",
    )

    run_clicked = st.button("Run Analysis", type="primary", disabled=not (acc_file and gyro_file))

    if not (acc_file and gyro_file):
        st.info("Upload both CSV files to enable analysis.")
        return

    if run_clicked:
        with st.spinner("Running pipeline: predicting windows → aggregating sets → scoring quality..."):
            try:
                # predict_from_raw_csv expects file paths, not in-memory
                # UploadedFile objects, so write them to a temp dir first.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    acc_path = Path(tmp_dir) / "session_acc.csv"
                    gyro_path = Path(tmp_dir) / "session_gyro.csv"
                    acc_path.write_bytes(acc_file.getvalue())
                    gyro_path.write_bytes(gyro_file.getvalue())

                    window_results = predict_from_raw_csv(str(acc_path), str(gyro_path))

                    # Load raw gyro for rep-counting (same format as predict_pipeline
                    # expects: no header, columns frame/timestamp_ms/x/y/z).
                    gyro_raw = pd.read_csv(
                        gyro_path, header=None,
                        names=["frame", "timestamp_ms", "x", "y", "z"]
                    )
                    raw_gyro_ts = (gyro_raw["timestamp_ms"] / 1000.0).to_numpy()
                    raw_gyro_xyz = gyro_raw[["x", "y", "z"]].to_numpy()

                    session_sets = aggregate_into_sets(
                        window_results,
                        raw_gyro_ts=raw_gyro_ts,
                        raw_gyro_xyz=raw_gyro_xyz,
                    )
                    scored = score_sets(session_sets)

                summary = session_summary(scored)

                # Save to DB so it shows up under Past Sessions too.
                session_id = f"{source_name or 'session'}_{datetime.now():%Y%m%d_%H%M%S}"
                save_session(scored, session_id=session_id, source_name=source_name or acc_file.name)

                st.success(f"Session saved as '{session_id}'.")

                # Stash in session_state so results persist across reruns
                # (e.g. if the user interacts with the table widget).
                reloaded = get_sets_for_session(session_id)
                st.session_state["last_scored_df"] = reloaded
                st.session_state["last_summary"] = summary
                st.session_state["last_session_label"] = session_id

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

    # Render whatever the most recent successful run produced, so results
    # stay visible even after Streamlit reruns the script on any widget interaction.
    if "last_scored_df" in st.session_state:
        render_results(
            st.session_state["last_scored_df"],
            st.session_state["last_summary"],
            st.session_state["last_session_label"],
        )


# =========================================================================
# Tab 2: Past Sessions
# =========================================================================

def past_sessions_tab():
    st.header("Browse Past Sessions")

    sessions_df = list_sessions()
    if sessions_df.empty:
        st.info("No sessions saved yet. Run a New Session analysis first.")
        return

    # Build a friendly label per session for the dropdown.
    def _label(row):
        return f"{row['session_id']}  —  {row['source_name']}  ({row['uploaded_at'][:19]} UTC)"

    sessions_df["_label"] = sessions_df.apply(_label, axis=1)
    choice = st.selectbox(
        "Select a session",
        options=sessions_df["_label"],
        index=0,
    )
    selected_session_id = sessions_df.loc[sessions_df["_label"] == choice, "session_id"].iloc[0]

    sets_df = get_sets_for_session(selected_session_id)
    if sets_df.empty:
        st.warning("This session has no sets recorded.")
        return

    summary = session_summary(sets_df)
    render_results(sets_df, summary, selected_session_id)


# =========================================================================
# Main
# =========================================================================

def main():
    st.title("🏋️ AI Personalized Rehabilitation Planner")
    st.caption("Phone-only IMU exercise recognition, set tracking, and quality feedback.")
    st.caption("DAY10-FIX-v2")

    tab1, tab2 = st.tabs(["📤 New Session", "🗂️ Past Sessions"])
    with tab1:
        new_session_tab()
    with tab2:
        past_sessions_tab()


if __name__ == "__main__":
    main()