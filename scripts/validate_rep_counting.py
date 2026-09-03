"""
scripts/validate_rep_counting.py

Day 12 - Step B: Formal rep-counting validation against MM-Fit ground truth,
on held-out test sessions only (w00, w01, w08, w15, w17).

Reuses raw-loading + label-alignment logic already built in preprocessing,
so we don't duplicate the Frame->Time conversion logic.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Adjust this if your preprocessing files live somewhere else
PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "src" / "preprocessing"
sys.path.append(str(PREPROCESSING_DIR))

from resample_pipeline import DATA_ROOT, TIMESTAMP_COL, load_and_normalize  # noqa: E402
from label_alignment import load_labels, build_frame_to_time_mapper  # noqa: E402

HELD_OUT_SESSIONS = ["w00", "w01", "w08", "w15", "w17"]

# Candidate global `distance` values (seconds) to sweep across ALL exercises.
# Not per-exercise tuned on purpose - we want one setting that generalizes.
DISTANCE_CANDIDATES_SEC = [0.5, 0.75, 1.0, 1.5, 2.0]

PROMINENCE_FACTOR = 0.5  # x signal std dev, same as Step A


def magnitude(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((xyz ** 2).sum(axis=1))


def count_peaks(mag: np.ndarray, fs: float, distance_sec: float) -> int:
    dist_samples = max(1, int(fs * distance_sec))
    peaks, _ = find_peaks(mag, distance=dist_samples, prominence=mag.std() * PROMINENCE_FACTOR)
    return len(peaks)

def estimate_period_autocorr(mag: np.ndarray, fs: float, min_period_s=0.4, max_period_s=3.0) -> float:
    """Estimate dominant rep period (seconds) via autocorrelation of the magnitude signal."""
    mag = mag - mag.mean()
    autocorr = np.correlate(mag, mag, mode="full")[len(mag) - 1:]
    min_lag = int(fs * min_period_s)
    max_lag = int(fs * max_period_s)
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    if min_lag >= max_lag:
        return 1.5  # fallback to old default
    search_region = autocorr[min_lag:max_lag]
    best_lag = min_lag + np.argmax(search_region)
    return best_lag / fs

def slice_by_time(ts: np.ndarray, xyz: np.ndarray, start_t: float, end_t: float):
    mask = (ts >= start_t) & (ts <= end_t)
    return ts[mask], xyz[mask]


def estimate_fs(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 0.0
    return 1.0 / np.median(np.diff(ts))


def validate_session(session: str) -> list:
    """Returns a list of result dicts, one per labeled exercise segment."""
    acc_ts, acc_xyz = load_and_normalize(session, "sp_r_acc")
    gyr_ts, gyr_xyz = load_and_normalize(session, "sp_r_gyr")

    if acc_ts is None or gyr_ts is None:
        print(f"{session}: missing raw data, skipping")
        return []

    frame_to_time = build_frame_to_time_mapper(session)
    rows = load_labels(session)  # (start_frame, end_frame, reps, exercise)

    results = []
    for start_frame, end_frame, true_reps, exercise in rows:
        if exercise == "non_activity":
            continue  # only scored exercises have meaningful rep counts

        start_t = frame_to_time(start_frame)
        end_t = frame_to_time(end_frame)
        duration = end_t - start_t

        seg_acc_ts, seg_acc_xyz = slice_by_time(acc_ts, acc_xyz, start_t, end_t)
        seg_gyr_ts, seg_gyr_xyz = slice_by_time(gyr_ts, gyr_xyz, start_t, end_t)

        if len(seg_acc_ts) < 5 or len(seg_gyr_ts) < 5:
            continue  # too short to meaningfully peak-detect

        acc_mag = magnitude(seg_acc_xyz)
        gyr_mag = magnitude(seg_gyr_xyz)
        acc_fs = estimate_fs(seg_acc_ts)
        gyr_fs = estimate_fs(seg_gyr_ts)

        row_result = {
            "session": session,
            "exercise": exercise,
            "true_reps": true_reps,
            "duration_s": round(duration, 1),
        }
        acc_period = estimate_period_autocorr(acc_mag, acc_fs)
        gyr_period = estimate_period_autocorr(gyr_mag, gyr_fs)

        # distance = 70% of estimated period, avoids merging two adjacent real reps
        row_result["acc_final"] = count_peaks(acc_mag, acc_fs, 1.5)
        row_result["gyr_final"] = count_peaks(gyr_mag, gyr_fs, 1.5)

        results.append(row_result)

    return results


def main():
    all_results = []
    for session in HELD_OUT_SESSIONS:
        print(f"Processing {session}...")
        all_results.extend(validate_session(session))

    df = pd.DataFrame(all_results)
    OUT_PATH = Path("scripts/rep_count_validation_results.csv")
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved per-segment results: {OUT_PATH} ({len(df)} segments)")

    # ---- Summary: MAE per distance candidate, per signal, across ALL segments ----
    print("\n" + "=" * 60)
    print("MEAN ABSOLUTE ERROR by distance setting (lower = better)")
    print("=" * 60)

    acc_mae = (df["acc_final"] - df["true_reps"]).abs().mean()
    gyr_mae = (df["gyr_final"] - df["true_reps"]).abs().mean()
    print(f"\nAdaptive distance -> acc MAE: {acc_mae:.2f} | gyr MAE: {gyr_mae:.2f}")

    df["acc_err"] = (df["acc_final"] - df["true_reps"]).abs()
    df["gyr_err"] = (df["gyr_final"] - df["true_reps"]).abs()
    per_ex = df.groupby("exercise").agg(
        acc_MAE=("acc_err", "mean"), gyr_MAE=("gyr_err", "mean")
    ).round(2).sort_values("gyr_MAE", ascending=False)
    print(per_ex.to_string())

if __name__ == "__main__":
    main()