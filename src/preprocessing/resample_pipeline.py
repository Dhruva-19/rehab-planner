"""
Day 2 - Step 3: Resampling (phone-only: sp_r_acc, sp_r_gyr)
---------------------------------------------------------------
Goal: turn the raw, slightly-irregular sensor readings into clean,
evenly-spaced chunks of data, ready for sliding-window feature extraction.

Two rules (explained in chat):
  1. Small timing gaps (jitter) -> smoothed via interpolation onto a fixed grid.
  2. Large gaps (real dropouts)  -> session is SPLIT into separate chunks here,
     never bridged with fake/interpolated data.

Input:  raw arrays with columns [Frame, Timestamp(ms), X, Y, Z]
Output: for each session, a list of chunks. Each chunk is a dict:
    {
        "start_time": float (seconds),
        "acc": np.ndarray of shape (T, 3)   # X, Y, Z, evenly sampled
        "gyr": np.ndarray of shape (T, 3)
        "timestamps": np.ndarray of shape (T,)  # seconds, evenly spaced
    }
"""

import numpy as np
from pathlib import Path

DATA_ROOT = Path("data/raw/mm-fit")
SESSIONS = [f"w{i:02d}" for i in range(21)]
TIMESTAMP_COL = 1
XYZ_COLS = slice(2, 5)

TARGET_HZ = 200.0          # confirmed actual rate from Step 2
GAP_THRESHOLD_S = 0.3      # gaps bigger than this (300ms) = real dropout, not jitter
MIN_CHUNK_DURATION_S = 2.0  # discard chunks shorter than this - too short to be useful


def load_and_normalize(session: str, modality: str):
    """Load a modality file and convert its timestamps from ms to seconds."""
    path = DATA_ROOT / session / f"{session}_{modality}.npy"
    if not path.exists():
        return None, None
    arr = np.load(path)
    timestamps = arr[:, TIMESTAMP_COL] / 1000.0  # ms -> seconds (confirmed in Step 2)
    xyz = arr[:, XYZ_COLS]
    return timestamps, xyz


def split_at_gaps(timestamps: np.ndarray, xyz: np.ndarray, gap_threshold_s: float):
    """
    Split a (timestamps, xyz) stream into a list of continuous segments,
    breaking wherever the gap between samples exceeds gap_threshold_s.
    """
    dt = np.diff(timestamps)
    gap_indices = np.where(dt > gap_threshold_s)[0]  # index of the sample BEFORE each gap

    # Segment boundaries: start of array, each gap point, end of array
    boundaries = [0] + list(gap_indices + 1) + [len(timestamps)]

    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 2:  # need at least 2 points to resample
            continue
        segments.append((timestamps[start:end], xyz[start:end]))
    return segments


def resample_segment(timestamps: np.ndarray, xyz: np.ndarray, target_hz: float):
    """
    Interpolate one continuous segment onto a uniform time grid at target_hz.
    Uses linear interpolation - fine for smoothing small jitter, since the
    true gap has already been ruled out by split_at_gaps().
    """
    start, end = timestamps[0], timestamps[-1]
    n_samples = int((end - start) * target_hz)
    if n_samples < 2:
        return None, None

    uniform_ts = np.linspace(start, end, n_samples)
    # Interpolate each axis (X, Y, Z) independently onto the uniform grid
    resampled_xyz = np.column_stack([
        np.interp(uniform_ts, timestamps, xyz[:, axis]) for axis in range(3)
    ])
    return uniform_ts, resampled_xyz


def process_session(session: str):
    """
    Full pipeline for one session:
      1. Load + normalize acc and gyro timestamps
      2. Split each into gap-free segments
      3. Resample each segment onto a uniform TARGET_HZ grid
      4. Keep only segments long enough to be useful
    Returns a list of chunk dicts (see module docstring).
    """
    acc_ts, acc_xyz = load_and_normalize(session, "sp_r_acc")
    gyr_ts, gyr_xyz = load_and_normalize(session, "sp_r_gyr")

    if acc_ts is None or gyr_ts is None:
        print(f"{session}: missing acc or gyro data, skipping")
        return []

    acc_segments = split_at_gaps(acc_ts, acc_xyz, GAP_THRESHOLD_S)

    chunks = []
    for seg_ts, seg_xyz in acc_segments:
        duration = seg_ts[-1] - seg_ts[0]
        if duration < MIN_CHUNK_DURATION_S:
            continue

        uniform_ts, acc_resampled = resample_segment(seg_ts, seg_xyz, TARGET_HZ)
        if uniform_ts is None:
            continue

        # Interpolate gyro onto this SAME uniform timeline (aligns acc + gyro)
        gyr_resampled = np.column_stack([
            np.interp(uniform_ts, gyr_ts, gyr_xyz[:, axis]) for axis in range(3)
        ])

        chunks.append({
            "start_time": uniform_ts[0],
            "timestamps": uniform_ts,
            "acc": acc_resampled,
            "gyr": gyr_resampled,
        })

    return chunks


if __name__ == "__main__":
    total_chunks = 0
    total_duration = 0.0

    for session in SESSIONS:
        chunks = process_session(session)
        session_duration = sum(c["timestamps"][-1] - c["timestamps"][0] for c in chunks)
        total_chunks += len(chunks)
        total_duration += session_duration
        print(f"{session}: {len(chunks)} chunks, {session_duration:.1f}s usable data")

    print(f"\nTotal: {total_chunks} chunks across all sessions, "
          f"{total_duration:.1f}s ({total_duration/60:.1f} min) of usable data")
