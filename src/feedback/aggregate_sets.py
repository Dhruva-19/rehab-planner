"""
src/feedback/aggregate_sets.py

Purpose: Convert noisy, overlapping per-window predictions (output of
predict_pipeline.predict_from_raw_csv) into clean, contiguous "exercise
sets" — the unit a rehab dashboard actually wants to show
("Set 1: bicep_curls, 0:12-0:34, 15 windows, avg confidence 0.91")
rather than a wall of per-window rows.

Day 8 update: added `confidence_std` and `raw_agreement` per set so the
quality/feedback engine has a stability signal to work with, on top of
the existing mean_confidence and is_short signals.

Day 12 update: added optional rep-counting per set, using gyroscope-
magnitude peak detection. distance=1.5s was locked after validating
against MM-Fit ground-truth reps on held-out test sessions (w00, w01,
w08, w15, w17) - gyro MAE 2.28 across 158 labeled segments, 10 exercise
types. Known limitation: less reliable for jumping_jacks and pushups,
where the hip-worn phone sees weaker/faster motion than at the limb
doing the work (see Day12_Summary.md for full breakdown).

Rep counting is OPT-IN: pass raw_gyro_ts + raw_gyro_xyz to get an
`estimated_reps` column; omit them and existing callers work unchanged
(estimated_reps will be NaN).
"""

from collections import Counter
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- Rep counting config (locked Day 12) ---
REP_COUNT_DISTANCE_SEC = 1.5  # gyro MAE 2.28 on held-out MM-Fit validation
REP_COUNT_PROMINENCE_FACTOR = 0.5


def _estimate_fs(ts: np.ndarray) -> float:
    """Median-based sampling rate estimate for an (irregularly-spaced) timestamp array."""
    if len(ts) < 2:
        return 0.0
    return 1.0 / np.median(np.diff(ts))


def count_reps_for_set(gyr_xyz: np.ndarray, fs: float) -> int:
    """
    Estimate rep count for one detected exercise set using gyroscope
    magnitude peak-counting. distance=1.5s locked from Day 12 validation
    (see module docstring). Returns 0 if the segment is too short or fs
    couldn't be estimated.
    """
    if len(gyr_xyz) < 5 or fs <= 0:
        return 0
    mag = np.sqrt((gyr_xyz ** 2).sum(axis=1))
    mag = mag - mag.mean()
    dist_samples = max(1, int(fs * REP_COUNT_DISTANCE_SEC))
    peaks, _ = find_peaks(
        mag,
        distance=dist_samples,
        prominence=mag.std() * REP_COUNT_PROMINENCE_FACTOR
    )
    return len(peaks)


def smooth_predictions(labels: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Rolling-mode smoothing over a sequence of labels.

    For each position i, replace label[i] with the most common label
    in the window [i - window_size//2, i + window_size//2] (clipped at
    the array edges). Ties are broken in favor of the ORIGINAL label at
    that position, so a genuine tie doesn't arbitrarily flip a
    confident prediction.

    window_size should be odd.
    """
    n = len(labels)
    half = window_size // 2
    smoothed = np.empty(n, dtype=labels.dtype)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts = Counter(labels[lo:hi])
        top_count = max(counts.values())
        candidates = [lbl for lbl, c in counts.items() if c == top_count]
        smoothed[i] = labels[i] if labels[i] in candidates else candidates[0]

    return smoothed


def aggregate_into_sets(results_df: pd.DataFrame,
                         smoothing_window: int = 5,
                         min_set_windows: int = 3,
                         gap_tolerance_factor: float = 1.5,
                         raw_gyro_ts: np.ndarray = None,
                         raw_gyro_xyz: np.ndarray = None) -> pd.DataFrame:
    """
    Turn per-window predictions into contiguous exercise "sets".

    Parameters
    ----------
    results_df : DataFrame from predict_pipeline.predict_from_raw_csv,
        columns: start_time, end_time, predicted_label,
        predicted_class_name, confidence.
    smoothing_window : odd int, rolling-mode kernel size.
    min_set_windows : sets with fewer windows than this are flagged
        `is_short=True` — not dropped or merged. Real transitions and
        genuinely short real sets both look like this; deciding which
        is which is not this function's job.
    gap_tolerance_factor : a start_time jump bigger than
        (median step * this factor) is treated as a real recording
        gap and always forces a new set, even if the label on both
        sides matches.
    raw_gyro_ts : optional, 1D array of raw gyroscope sample timestamps
        (seconds), same clock as results_df's start_time/end_time.
        Required (together with raw_gyro_xyz) to compute estimated_reps.
    raw_gyro_xyz : optional, (N, 3) array of raw gyroscope x/y/z samples,
        aligned index-for-index with raw_gyro_ts.

    Returns
    -------
    DataFrame, one row per set:
        label, start_time, end_time, duration_s, num_windows,
        mean_confidence, confidence_std, raw_agreement, is_short,
        estimated_reps

        confidence_std   : std of per-window confidence within the set.
                            NaN for single-window sets -> treat as 0
                            (no variability observed) downstream.
        raw_agreement    : fraction of windows whose RAW (pre-smoothing)
                            predicted_class_name matches the set's final
                            smoothed label. 1.0 = every window already
                            agreed before smoothing stepped in (stable,
                            confident run). Lower values mean smoothing
                            had to overrule a chunk of flickering raw
                            predictions to produce this set — a proxy
                            for jerky/inconsistent movement.
        estimated_reps   : gyro-magnitude peak count for this set's time
                            range. NaN if raw_gyro_ts/raw_gyro_xyz were
                            not supplied, or if label is "non_activity"
                            (rep count is meaningless for rest periods).
    """
    empty_cols = ["label", "start_time", "end_time", "duration_s",
                  "num_windows", "mean_confidence", "confidence_std",
                  "raw_agreement", "is_short", "estimated_reps"]
    if results_df.empty:
        return pd.DataFrame(columns=empty_cols)

    df = results_df.sort_values("start_time").reset_index(drop=True)

    # --- 1. Smooth label flicker ---
    raw_labels = df["predicted_class_name"].to_numpy()
    smoothed_labels = smooth_predictions(raw_labels, window_size=smoothing_window)

    # --- 2. Detect real chunk/session gaps from window spacing ---
    start_times = df["start_time"].to_numpy()
    steps = np.diff(start_times)
    if len(steps) > 0:
        expected_step = np.median(steps)
        is_gap = np.concatenate([[False], steps > expected_step * gap_tolerance_factor])
    else:
        is_gap = np.array([False])

    # --- 3. Run-length encode: new set on label change OR a real gap ---
    label_changed = np.concatenate([[True], smoothed_labels[1:] != smoothed_labels[:-1]])
    new_set_boundary = label_changed | is_gap
    set_id = np.cumsum(new_set_boundary)

    df = df.assign(smoothed_label=smoothed_labels, set_id=set_id)

    # Row-level agreement flag: did this window's raw prediction match
    # the final smoothed label it ended up contributing to? Computed
    # here (row level) so the groupby below can just take the mean.
    df["_agrees"] = (df["predicted_class_name"] == df["smoothed_label"]).astype(float)

    # --- 4. Aggregate each set ---
    sets = (
        df.groupby("set_id")
        .agg(
            label=("smoothed_label", "first"),
            start_time=("start_time", "min"),
            end_time=("end_time", "max"),
            num_windows=("start_time", "count"),
            mean_confidence=("confidence", "mean"),
            confidence_std=("confidence", "std"),
            raw_agreement=("_agrees", "mean"),
        )
        .reset_index(drop=True)
    )
    sets["duration_s"] = sets["end_time"] - sets["start_time"]
    sets["is_short"] = sets["num_windows"] < min_set_windows

    # --- 5. Rep counting (optional, needs raw gyro) ---
    if raw_gyro_ts is not None and raw_gyro_xyz is not None:
        reps = []
        for _, row in sets.iterrows():
            mask = (raw_gyro_ts >= row["start_time"]) & (raw_gyro_ts <= row["end_time"])
            seg_xyz = raw_gyro_xyz[mask]
            seg_ts = raw_gyro_ts[mask]
            fs = _estimate_fs(seg_ts)
            reps.append(count_reps_for_set(seg_xyz, fs))
        sets["estimated_reps"] = reps
    else:
        sets["estimated_reps"] = np.nan

    # Rep count is meaningless for rest periods
    sets.loc[sets["label"] == "non_activity", "estimated_reps"] = np.nan

    return sets[empty_cols]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "inference"))
    from predict_pipeline import predict_from_raw_csv

    ACC_CSV = "data/sample_upload/session_acc.csv"
    GYRO_CSV = "data/sample_upload/session_gyro.csv"

    window_results = predict_from_raw_csv(ACC_CSV, GYRO_CSV)
    session_sets = aggregate_into_sets(window_results)

    print(f"\n{len(window_results)} raw windows -> {len(session_sets)} sets\n")
    print(session_sets.to_string(index=False))