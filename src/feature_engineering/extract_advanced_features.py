"""
src/feature_engineering/extract_advanced_features.py

Purpose: Compute rhythm/tempo-based features (jerk, autocorrelation
periodicity, peak cadence) that the original 77 statistical features
don't capture, then merge them into the existing features.csv.

Motivation: tricep_extensions, dumbbell_shoulder_press, and
lateral_shoulder_raises (classes 5, 8, 9) all have similar amplitude
statistics but differ in movement TEMPO and SMOOTHNESS — these new
features target exactly that.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# -----------------------------
# Config
# -----------------------------
WINDOWS_PATH = "data/processed/labeled_windows.npz"
EXISTING_FEATURES_PATH = "data/processed/features.csv"
OUTPUT_PATH = "data/processed/features_v2.csv"

SAMPLING_RATE_HZ = 200
CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
EPS = 1e-8


def compute_jerk_features(signal: np.ndarray):
    """
    Jerk = rate of change of acceleration (or gyro rate, for gyro channels).
    High jerk = abrupt, jarring movement. Low jerk = smooth, controlled movement.
    """
    jerk = np.diff(signal) * SAMPLING_RATE_HZ
    jerk_mean_abs = np.mean(np.abs(jerk))
    jerk_std = np.std(jerk)
    return float(jerk_mean_abs), float(jerk_std)


def compute_autocorr_features(signal: np.ndarray):
    """
    Autocorrelation: correlate the signal with a time-shifted copy of itself.
    The lag with the highest correlation (after skipping trivial near-zero
    lags) tells us the dominant repetition period -> converts to rep rate (Hz).
    The correlation STRENGTH at that peak tells us how "clean"/periodic the
    movement is (a strong, regular rep pattern vs. noisy/irregular motion).
    """
    signal = signal - np.mean(signal)
    n = len(signal)

    if np.std(signal) < EPS:
        return 0.0, 0.0  # flat/zero-variance signal, nothing periodic to find

    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[n - 1:]              # keep lag=0 onward
    autocorr = autocorr / (autocorr[0] + EPS)  # normalize so lag=0 -> 1.0

    min_lag = int(SAMPLING_RATE_HZ * 0.2)     # ignore lags < 0.2s (trivial peak near 0)
    if min_lag >= len(autocorr) - 1:
        return 0.0, 0.0

    search_region = autocorr[min_lag:]
    peak_idx = int(np.argmax(search_region))
    peak_strength = float(search_region[peak_idx])
    peak_lag_samples = peak_idx + min_lag

    rep_rate_hz = SAMPLING_RATE_HZ / peak_lag_samples if peak_lag_samples > 0 else 0.0
    return peak_strength, float(rep_rate_hz)


def compute_peak_features(signal: np.ndarray):
    """
    Count distinct "rep peaks" in the window and measure how REGULAR the
    spacing between them is (low variability = consistent rep cadence,
    high variability = irregular/non-rhythmic movement, closer to rest).
    """
    signal = signal - np.mean(signal)
    std = np.std(signal)

    if std < EPS:
        return 0, 0.0

    height_thresh = 0.3 * std
    min_distance = int(SAMPLING_RATE_HZ * 0.15)  # peaks must be >= 0.15s apart
    peaks, _ = find_peaks(signal, height=height_thresh, distance=min_distance)

    peak_count = len(peaks)
    if peak_count >= 2:
        intervals = np.diff(peaks) / SAMPLING_RATE_HZ
        interval_cv = float(np.std(intervals) / (np.mean(intervals) + EPS))
    else:
        interval_cv = 0.0

    return peak_count, interval_cv


def extract_advanced_features_single_window(window: np.ndarray) -> dict:
    """window shape: (600, 6) -> one 3-second window, 6 IMU channels."""
    feats = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = window[:, ch_idx]

        jerk_mean_abs, jerk_std = compute_jerk_features(signal)
        autocorr_strength, rep_rate_hz = compute_autocorr_features(signal)
        peak_count, interval_cv = compute_peak_features(signal)

        feats[f"{ch_name}_jerk_mean_abs"] = jerk_mean_abs
        feats[f"{ch_name}_jerk_std"] = jerk_std
        feats[f"{ch_name}_autocorr_peak_strength"] = autocorr_strength
        feats[f"{ch_name}_rep_rate_hz"] = rep_rate_hz
        feats[f"{ch_name}_peak_count"] = peak_count
        feats[f"{ch_name}_peak_interval_cv"] = interval_cv

    return feats


def extract_all(X: np.ndarray, y: np.ndarray, sessions: np.ndarray) -> pd.DataFrame:
    """Loop over every window and build the advanced feature table."""
    rows = []
    n = X.shape[0]

    for i in range(n):
        feats = extract_advanced_features_single_window(X[i])
        feats["label"] = y[i]
        feats["session"] = sessions[i]
        rows.append(feats)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n} windows...")

    return pd.DataFrame(rows)


def merge_with_existing(df_advanced: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    """
    Merge new features into the existing features.csv, column-wise.
    Includes safety checks so a row-order mismatch fails loudly instead
    of silently corrupting your dataset.
    """
    df_existing = pd.read_csv(existing_path)

    if len(df_existing) != len(df_advanced):
        raise ValueError(
            f"Row count mismatch: existing features.csv has {len(df_existing)} rows, "
            f"advanced features have {len(df_advanced)} rows. Both must come from "
            f"the SAME labeled_windows.npz, in the SAME order."
        )

    mismatches = int((df_existing["label"].values != df_advanced["label"].values).sum())
    if mismatches > 0:
        raise ValueError(
            f"{mismatches} label mismatches between existing and advanced features — "
            f"row order does not align. Aborting merge to avoid corrupting data."
        )

    new_cols = [c for c in df_advanced.columns if c not in ("label", "session")]
    df_merged = pd.concat([df_existing.reset_index(drop=True),
                            df_advanced[new_cols].reset_index(drop=True)], axis=1)
    return df_merged


if __name__ == "__main__":
    print("Loading labeled windows...")
    data = np.load(WINDOWS_PATH, allow_pickle=True)
    X, y, sessions = data["X"], data["y"], data["sessions"]
    print(f"X shape: {X.shape}")

    print("\nExtracting advanced rhythm/jerk features...")
    df_advanced = extract_all(X, y, sessions)
    print(f"Advanced features shape: {df_advanced.shape}")

    print("\nMerging with existing features.csv...")
    df_merged = merge_with_existing(df_advanced, EXISTING_FEATURES_PATH)
    print(f"Merged features shape: {df_merged.shape}")

    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged features to {OUTPUT_PATH}")