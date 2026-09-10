"""
compare_orientation_axes.py

Diagnostic: compares per-axis (not magnitude) accel/gyro statistics
between a genuine MM-Fit lateral_shoulder_raises training window and
the live-captured lateral raises session, to check for an axis
orientation mismatch (phone mounted differently than MM-Fit's sp_r
convention).

Magnitude-based checks are orientation-invariant by design and can't
catch this -- this script looks at x/y/z individually.

Usage:
    python compare_orientation_axes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
LABELED_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_windows.npz"
SAMPLE_UPLOAD_DIR = PROJECT_ROOT / "data" / "sample_upload"

LABEL_NAMES = {
    0: "squats", 1: "lunges", 2: "bicep_curls", 3: "situps", 4: "pushups",
    5: "tricep_extensions", 6: "dumbbell_rows", 7: "jumping_jacks",
    8: "dumbbell_shoulder_press", 9: "lateral_shoulder_raises",
    10: "non_activity",
}
LATERAL_RAISES_LABEL = 9
BICEP_CURLS_LABEL = 2


def summarize_axes(xyz: np.ndarray, label: str):
    print(f"\n--- {label} ---")
    for i, axis in enumerate(["x", "y", "z"]):
        col = xyz[:, i]
        print(f"  {axis}:  mean={col.mean():7.3f}  std={col.std():6.3f}  "
              f"min={col.min():7.3f}  max={col.max():7.3f}")


def main():
    # --- 1. Load real MM-Fit training windows for lateral_shoulder_raises ---
    data = np.load(LABELED_WINDOWS_PATH, allow_pickle=True)
    X, y = data["X"], data["y"]  # X: (N, 600, 6) -> cols [acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z]

    lateral_idx = np.where(y == LATERAL_RAISES_LABEL)[0]
    bicep_idx = np.where(y == BICEP_CURLS_LABEL)[0]

    print(f"Found {len(lateral_idx)} lateral_shoulder_raises windows and "
          f"{len(bicep_idx)} bicep_curls windows in training data.\n")

    # Average across all training windows of each class, per-axis
    lateral_acc = X[lateral_idx][:, :, 0:3].reshape(-1, 3)
    lateral_gyr = X[lateral_idx][:, :, 3:6].reshape(-1, 3)
    bicep_acc = X[bicep_idx][:, :, 0:3].reshape(-1, 3)
    bicep_gyr = X[bicep_idx][:, :, 3:6].reshape(-1, 3)

    print("=" * 60)
    print("TRAINING DATA (MM-Fit sp_r, ground truth orientation)")
    print("=" * 60)
    summarize_axes(lateral_acc, "lateral_shoulder_raises -- accel (m/s^2)")
    summarize_axes(lateral_gyr, "lateral_shoulder_raises -- gyro (rad/s)")
    summarize_axes(bicep_acc, "bicep_curls -- accel (m/s^2)")
    summarize_axes(bicep_gyr, "bicep_curls -- gyro (rad/s)")

    # --- 2. Load live capture, isolate the active segment (~t=32-49s) ---
    acc_df = pd.read_csv(SAMPLE_UPLOAD_DIR / "Lateral_raises_acc.csv",
                          header=None, names=["frame", "timestamp_ms", "x", "y", "z"])
    gyro_df = pd.read_csv(SAMPLE_UPLOAD_DIR / "Lateral_raises_gyro.csv",
                           header=None, names=["frame", "timestamp_ms", "x", "y", "z"])

    acc_ts = acc_df["timestamp_ms"].to_numpy() / 1000.0
    gyro_ts = gyro_df["timestamp_ms"].to_numpy() / 1000.0
    acc_ts -= acc_ts[0]
    gyro_ts -= gyro_ts[0]

    ACTIVE_START, ACTIVE_END = 32.0, 49.0
    acc_mask = (acc_ts >= ACTIVE_START) & (acc_ts <= ACTIVE_END)
    gyro_mask = (gyro_ts >= ACTIVE_START) & (gyro_ts <= ACTIVE_END)

    live_acc = acc_df.loc[acc_mask, ["x", "y", "z"]].to_numpy()
    live_gyr = gyro_df.loc[gyro_mask, ["x", "y", "z"]].to_numpy()

    print("\n" + "=" * 60)
    print(f"LIVE CAPTURE (active segment, t={ACTIVE_START}-{ACTIVE_END}s)")
    print("=" * 60)
    summarize_axes(live_acc, "live lateral raises -- accel (m/s^2)")
    summarize_axes(live_gyr, "live lateral raises -- gyro (rad/s)")

    print("\n--- What to look for ---")
    print("  Compare the LIVE accel/gyro axis pattern above against BOTH")
    print("  training classes. If live axes look more like TRAINING")
    print("  bicep_curls than TRAINING lateral_shoulder_raises (e.g. the")
    print("  axis carrying the largest std/range doesn't match, or signs")
    print("  are flipped), that's a strong signal of an axis mapping /")
    print("  orientation mismatch between live capture and MM-Fit's sp_r")
    print("  mounting convention -- not a model or feature-extraction bug.")


if __name__ == "__main__":
    main()