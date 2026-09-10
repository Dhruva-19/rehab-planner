"""
scripts/sweep_rep_counting.py

Day 19 - Per-exercise distance sweep for rep counting.

Extends Day 12's validate_rep_counting.py: instead of scoring a single
hardcoded distance=1.5s, this sweeps a range of candidate distances per
segment and saves LONG-format results (one row per segment x distance)
so we can find each exercise's own MAE-minimizing distance, without
touching the original Day 12 validation script/output.

Same session set, same loading/label-alignment logic as Day 12, so
results stay comparable to the locked 2.28 gyro MAE baseline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "src" / "preprocessing"
sys.path.append(str(PREPROCESSING_DIR))

from resample_pipeline import load_and_normalize  # noqa: E402
from label_alignment import load_labels, build_frame_to_time_mapper  # noqa: E402

HELD_OUT_SESSIONS = ["w00", "w01", "w08", "w15", "w17"]

# Finer sweep than Day 12's 5-point check, since we're now looking for
# per-exercise minima rather than one global value.
DISTANCE_CANDIDATES_SEC = [round(x, 2) for x in np.arange(0.4, 2.01, 0.1)]

PROMINENCE_FACTOR = 0.5  # unchanged from Day 12


def magnitude(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((xyz ** 2).sum(axis=1))


def count_peaks(mag: np.ndarray, fs: float, distance_sec: float) -> int:
    dist_samples = max(1, int(fs * distance_sec))
    peaks, _ = find_peaks(mag, distance=dist_samples, prominence=mag.std() * PROMINENCE_FACTOR)
    return len(peaks)


def slice_by_time(ts: np.ndarray, xyz: np.ndarray, start_t: float, end_t: float):
    mask = (ts >= start_t) & (ts <= end_t)
    return ts[mask], xyz[mask]


def estimate_fs(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 0.0
    return 1.0 / np.median(np.diff(ts))


def sweep_session(session: str) -> list:
    """Returns a list of result dicts: one per (segment, distance candidate)."""
    acc_ts, acc_xyz = load_and_normalize(session, "sp_r_acc")
    gyr_ts, gyr_xyz = load_and_normalize(session, "sp_r_gyr")

    if acc_ts is None or gyr_ts is None:
        print(f"{session}: missing raw data, skipping")
        return []

    frame_to_time = build_frame_to_time_mapper(session)
    rows = load_labels(session)

    results = []
    for start_frame, end_frame, true_reps, exercise in rows:
        if exercise == "non_activity":
            continue

        start_t = frame_to_time(start_frame)
        end_t = frame_to_time(end_frame)

        seg_acc_ts, seg_acc_xyz = slice_by_time(acc_ts, acc_xyz, start_t, end_t)
        seg_gyr_ts, seg_gyr_xyz = slice_by_time(gyr_ts, gyr_xyz, start_t, end_t)

        if len(seg_acc_ts) < 5 or len(seg_gyr_ts) < 5:
            continue

        acc_mag = magnitude(seg_acc_xyz)
        gyr_mag = magnitude(seg_gyr_xyz)
        acc_fs = estimate_fs(seg_acc_ts)
        gyr_fs = estimate_fs(seg_gyr_ts)

        # segment identity, so we can pivot / group later
        seg_id = f"{session}_{exercise}_{start_frame}_{end_frame}"

        for d in DISTANCE_CANDIDATES_SEC:
            results.append({
                "seg_id": seg_id,
                "session": session,
                "exercise": exercise,
                "true_reps": true_reps,
                "distance_sec": d,
                "acc_reps": count_peaks(acc_mag, acc_fs, d),
                "gyr_reps": count_peaks(gyr_mag, gyr_fs, d),
            })

    return results


def main():
    all_results = []
    for session in HELD_OUT_SESSIONS:
        print(f"Processing {session}...")
        all_results.extend(sweep_session(session))

    df = pd.DataFrame(all_results)
    df["acc_err"] = (df["acc_reps"] - df["true_reps"]).abs()
    df["gyr_err"] = (df["gyr_reps"] - df["true_reps"]).abs()

    OUT_PATH = Path("scripts/rep_count_sweep_results.csv")
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved long-format sweep results: {OUT_PATH} ({len(df)} rows)")

    # ---- Per exercise x distance MAE table (gyro = primary signal, per Day 12) ----
    pivot_gyr = df.groupby(["exercise", "distance_sec"])["gyr_err"].mean().unstack("distance_sec").round(2)
    pivot_gyr.to_csv("scripts/rep_count_sweep_per_exercise_gyr.csv")

    print("\n" + "=" * 70)
    print("GYRO MAE per exercise x distance (rows=exercise, cols=distance_sec)")
    print("=" * 70)
    print(pivot_gyr.to_string())

    # ---- Best distance per exercise, and what MAE it would achieve ----
    best_per_exercise = pivot_gyr.idxmin(axis=1).to_frame("best_distance_sec")
    best_per_exercise["best_gyr_MAE"] = pivot_gyr.min(axis=1)
    best_per_exercise["mae_at_1.5s"] = pivot_gyr[1.5] if 1.5 in pivot_gyr.columns else np.nan
    best_per_exercise["improvement"] = best_per_exercise["mae_at_1.5s"] - best_per_exercise["best_gyr_MAE"]
    best_per_exercise = best_per_exercise.sort_values("improvement", ascending=False)

    print("\n" + "=" * 70)
    print("BEST DISTANCE PER EXERCISE vs current locked 1.5s")
    print("=" * 70)
    print(best_per_exercise.to_string())

    # ---- Sanity check: pooled MAE at 1.5s should match Day 12's 2.28 ----
    pooled_at_1_5 = df[df["distance_sec"] == 1.5]["gyr_err"].mean()
    print(f"\nPooled gyro MAE at distance=1.5s (sanity check vs Day12's 2.28): {pooled_at_1_5:.2f}")


if __name__ == "__main__":
    main()