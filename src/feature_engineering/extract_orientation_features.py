"""
src/feature_engineering/extract_orientation_features.py

Purpose: Compute tilt/orientation-angle features from the accelerometer,
targeting the remaining confusion between tricep_extensions (5),
dumbbell_shoulder_press (8), and lateral_shoulder_raises (9).

Motivation: rhythm/jerk features (v2) capture TEMPO, but these three
exercises can share similar tempo while moving through different PLANES
of motion (overhead vs. lateral vs. elbow-localized). The angle between
accelerometer axis pairs approximates wrist tilt/orientation, which
differs by exercise plane even when amplitude and rhythm look similar.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
WINDOWS_PATH = "data/processed/labeled_windows.npz"
EXISTING_FEATURES_PATH = "data/processed/features_v2.csv"
OUTPUT_PATH = "data/processed/features_v3.csv"

# Accelerometer channel indices in the (600, 6) window array
ACC_X, ACC_Y, ACC_Z = 0, 1, 2
EPS = 1e-8


def compute_angle_series(axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    """
    Angle (in radians) between two accelerometer axes at every timestep.
    As the wrist tilts, gravity's projection onto these two axes shifts,
    so this angle traces the "orientation path" of the arm through the window.
    np.unwrap prevents artificial jumps when the angle crosses +-pi.
    """
    angle = np.arctan2(axis_b, axis_a)
    return np.unwrap(angle)


def compute_orientation_features_single_window(window: np.ndarray) -> dict:
    """window shape: (600, 6) -> one 3-second window."""
    acc_x = window[:, ACC_X]
    acc_y = window[:, ACC_Y]
    acc_z = window[:, ACC_Z]

    axis_pairs = {
        "xy": (acc_x, acc_y),
        "yz": (acc_y, acc_z),
        "xz": (acc_x, acc_z),
    }

    feats = {}
    for pair_name, (a, b) in axis_pairs.items():
        angle_series = compute_angle_series(a, b)

        angle_range = float(np.max(angle_series) - np.min(angle_series))
        angle_std = float(np.std(angle_series))

        feats[f"tilt_{pair_name}_range"] = angle_range
        feats[f"tilt_{pair_name}_std"] = angle_std

    return feats


def extract_all(X: np.ndarray, y: np.ndarray, sessions: np.ndarray) -> pd.DataFrame:
    """Loop over every window and build the orientation feature table."""
    rows = []
    n = X.shape[0]

    for i in range(n):
        feats = compute_orientation_features_single_window(X[i])
        feats["label"] = y[i]
        feats["session"] = sessions[i]
        rows.append(feats)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{n} windows...")

    return pd.DataFrame(rows)


def merge_with_existing(df_new: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    """
    Merge new features into features_v2.csv, with the same row-alignment
    safety checks used in the previous merge step.
    """
    df_existing = pd.read_csv(existing_path)

    if len(df_existing) != len(df_new):
        raise ValueError(
            f"Row count mismatch: existing has {len(df_existing)} rows, "
            f"new features have {len(df_new)} rows."
        )

    mismatches = int((df_existing["label"].values != df_new["label"].values).sum())
    if mismatches > 0:
        raise ValueError(
            f"{mismatches} label mismatches — row order does not align. "
            f"Aborting merge to avoid corrupting data."
        )

    new_cols = [c for c in df_new.columns if c not in ("label", "session")]
    df_merged = pd.concat([df_existing.reset_index(drop=True),
                            df_new[new_cols].reset_index(drop=True)], axis=1)
    return df_merged


if __name__ == "__main__":
    print("Loading labeled windows...")
    data = np.load(WINDOWS_PATH, allow_pickle=True)
    X, y, sessions = data["X"], data["y"], data["sessions"]
    print(f"X shape: {X.shape}")

    print("\nExtracting orientation/tilt features...")
    df_orientation = extract_all(X, y, sessions)
    print(f"Orientation features shape: {df_orientation.shape}")

    print("\nMerging with existing features_v2.csv...")
    df_merged = merge_with_existing(df_orientation, EXISTING_FEATURES_PATH)
    print(f"Merged features shape: {df_merged.shape}")

    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged features to {OUTPUT_PATH}")