"""
Day 3 - Step 1: Label Alignment (phone-only: sp_r_acc + sp_r_gyr windows)
---------------------------------------------------------------------------
Goal: attach an exercise label to every (600, 6) window produced in Day 2.

The problem: labels.csv gives exercise segments as [start_frame, end_frame],
but our windows only carry [start_time, end_time] in seconds (Frame numbers
were dropped during resampling). To bridge this, we rebuild a Frame -> Time
lookup straight from each session's raw sp_r_acc.npy file (columns 0 and 1
are Frame and Timestamp for the exact same samples), convert every labeled
segment into a time interval, and then match by time overlap against each
window.

Labeling rule: majority vote by time-overlap, with a purity threshold.
  - For a window, sum up how many seconds fall inside each exercise segment.
  - Any time NOT covered by a labeled segment counts as "non_activity".
  - The window's label = whichever class covers the most time.
  - purity = (time in majority class) / (window duration)
  - Windows with purity below PURITY_THRESHOLD are discarded as ambiguous
    (they straddle a transition too evenly to trust).

Run from your project root (same folder as resample_pipeline.py / sliding_windows.py).
"""

import numpy as np
from pathlib import Path
from collections import Counter

from resample_pipeline import DATA_ROOT, TIMESTAMP_COL
from sliding_windows import process_all_sessions

PURITY_THRESHOLD = 0.75  # keep a window only if >=75% of it is one class

ACTIONS = [
    "squats", "lunges", "bicep_curls", "situps", "pushups",
    "tricep_extensions", "dumbbell_rows", "jumping_jacks",
    "dumbbell_shoulder_press", "lateral_shoulder_raises", "non_activity",
]


def load_labels(session: str) -> list:
    """
    Load a session's labels.csv (no header). Each row:
        start_frame, end_frame, reps, exercise_name
    """
    path = DATA_ROOT / session / f"{session}_labels.csv"
    if not path.exists():
        return []

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start_frame, end_frame, reps, exercise = line.split(",")
            rows.append((float(start_frame), float(end_frame), int(reps), exercise))
    return rows


def build_frame_to_time_mapper(session: str):
    """
    Build a Frame -> Timestamp(seconds) converter for one session, using the
    RAW sp_r_acc.npy file directly (before resampling touched it). Frame and
    Timestamp are columns 0 and 1 of the same array, so this is a genuine
    1:1 lookup, not an approximation.
    """
    path = DATA_ROOT / session / f"{session}_sp_r_acc.npy"
    arr = np.load(path)

    frames = arr[:, 0]
    times_s = arr[:, TIMESTAMP_COL] / 1000.0  # ms -> seconds, same convention as resample_pipeline

    # Defensive: np.interp requires the x-values (frames) to be sorted ascending.
    order = np.argsort(frames)
    frames_sorted = frames[order]
    times_sorted = times_s[order]

    def frame_to_time(frame_value):
        return float(np.interp(frame_value, frames_sorted, times_sorted))

    return frame_to_time


def labels_to_time_intervals(session: str) -> list:
    """
    Convert a session's labels.csv (frame-based) into a sorted list of
    (start_time_s, end_time_s, exercise_name) tuples.
    """
    frame_to_time = build_frame_to_time_mapper(session)
    rows = load_labels(session)

    intervals = []
    for start_frame, end_frame, reps, exercise in rows:
        start_t = frame_to_time(start_frame)
        end_t = frame_to_time(end_frame)
        intervals.append((start_t, end_t, exercise))

    intervals.sort(key=lambda r: r[0])
    return intervals


def _overlap_seconds(a_start, a_end, b_start, b_end) -> float:
    """Overlap duration (s) between two closed intervals; 0 if no overlap."""
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, hi - lo)


def label_window(window: dict, intervals: list):
    """
    Majority-vote label a single window against this session's labeled
    intervals. Returns (label, purity).
    """
    w_start, w_end = window["start_time"], window["end_time"]
    w_duration = w_end - w_start

    class_time = Counter()
    covered = 0.0

    for i_start, i_end, exercise in intervals:
        if i_end < w_start:
            continue  # this labeled segment ends before the window starts
        if i_start > w_end:
            break  # intervals are sorted by start_time -> nothing further can overlap
        ov = _overlap_seconds(w_start, w_end, i_start, i_end)
        if ov > 0:
            class_time[exercise] += ov
            covered += ov

    # Any time not explicitly covered by a labeled exercise segment is rest/non_activity
    class_time["non_activity"] += max(0.0, w_duration - covered)

    majority_label, majority_time = class_time.most_common(1)[0]
    purity = majority_time / w_duration if w_duration > 0 else 0.0
    return majority_label, purity


def align_session(session: str, session_windows: list) -> list:
    """Attach a label + purity score to every window in one session."""
    intervals = labels_to_time_intervals(session)
    labeled = []
    for window in session_windows:
        label, purity = label_window(window, intervals)
        labeled.append({
            "data": window["data"],       # (600, 6) sensor array, unchanged
            "label": label,
            "purity": purity,
            "start_time": window["start_time"],
            "end_time": window["end_time"],
        })
    return labeled


def main():
    all_windows = process_all_sessions()  # {session: [windows]} from Day 2

    all_labeled = []
    kept, discarded = 0, 0
    class_counts = Counter()

    for session, session_windows in all_windows.items():
        labeled = align_session(session, session_windows)
        session_kept = 0
        for w in labeled:
            if w["purity"] >= PURITY_THRESHOLD:
                kept += 1
                session_kept += 1
                class_counts[w["label"]] += 1
                all_labeled.append(w)
            else:
                discarded += 1
        print(f"{session}: {len(labeled)} windows -> {session_kept} kept "
              f"(purity >= {PURITY_THRESHOLD})")

    print(f"\nTOTAL: {kept} windows kept, {discarded} discarded as ambiguous "
          f"(purity < {PURITY_THRESHOLD})")
    print(f"Kept ratio: {kept / (kept + discarded):.1%}")

    print("\nClass distribution (kept windows only):")
    for cls, count in class_counts.most_common():
        print(f"  {cls:<28} {count:>6}")

    return all_labeled


if __name__ == "__main__":
    main()
