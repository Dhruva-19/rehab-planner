"""
Day 2 - Step 4: Sliding Windows (phone-only: sp_r_acc + sp_r_gyr combined)
-----------------------------------------------------------------------------
Goal: slice each clean, gap-free chunk (from Step 3) into fixed-length,
overlapping windows - the actual unit a model will be trained on.

NOTE: This script does NOT assign exercise labels yet - that requires
inspecting the labels.csv format first, which we haven't done. Labels get
attached in the next step. For now, this just confirms the windowing
mechanics work and tells us how many training examples we'll end up with.

Run from your project root (same folder as resample_pipeline.py).
"""

import numpy as np
from resample_pipeline import process_session, SESSIONS, TARGET_HZ

WINDOW_SECONDS = 3.0
OVERLAP_FRACTION = 0.5  # 50% overlap between consecutive windows

WINDOW_SIZE = int(WINDOW_SECONDS * TARGET_HZ)          # 600 samples
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_FRACTION))  # 300 samples


def slide_windows_over_chunk(chunk: dict) -> list:
    """
    Slide a fixed-size window across one clean chunk.

    Combines acc (3 columns: X,Y,Z) and gyro (3 columns: X,Y,Z) into a
    single (WINDOW_SIZE, 6) array per window - this is the raw feature
    matrix later steps will compute statistics from.
    """
    combined = np.hstack([chunk["acc"], chunk["gyr"]])  # shape: (T, 6)
    timestamps = chunk["timestamps"]
    n_samples = len(combined)

    windows = []
    for start in range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        windows.append({
            "data": combined[start:end],       # shape (600, 6)
            "start_time": timestamps[start],
            "end_time": timestamps[end - 1],
        })
    return windows


def process_all_sessions() -> dict:
    """Run resampling + windowing for every session. Returns {session: [windows]}."""
    all_windows = {}
    for session in SESSIONS:
        chunks = process_session(session)
        session_windows = []
        for chunk in chunks:
            session_windows.extend(slide_windows_over_chunk(chunk))
        all_windows[session] = session_windows
    return all_windows


if __name__ == "__main__":
    all_windows = process_all_sessions()

    total_windows = 0
    for session, windows in all_windows.items():
        print(f"{session}: {len(windows)} windows")
        total_windows += len(windows)

    print(f"\nTotal windows across all sessions: {total_windows}")
    print(f"Each window shape: ({WINDOW_SIZE}, 6)  # 600 samples x [accX,accY,accZ,gyrX,gyrY,gyrZ]")
