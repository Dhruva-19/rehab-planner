"""
Diagnostic: inspect a raw live-capture CSV pair (acc + gyro) saved by
backend/main.py's /ingest endpoint, to check for issues specific to
browser DeviceMotion capture that wouldn't show up in MM-Fit's own data.

Usage (from project root, inside the venv that has pandas/numpy):
    python inspect_live_raw.py <session_name>

Example:
    python inspect_live_raw.py Lateral_raises_20260908_162902

This looks for the files:
    data/sample_upload/<session_name>_acc.csv
    data/sample_upload/<session_name>_gyro.csv

What it checks and why:
  1. Effective sample rate + jitter (std of inter-sample gaps).
     DeviceMotion is event-driven, not a fixed-rate hardware timer -
     if gaps are wildly uneven, your gap-aware resampler may be
     interpolating over irregular holes differently than it does on
     MM-Fit's steadier native captures.
  2. Count of "large" gaps (>100ms) - these are exactly what a
     gap-aware resampler treats specially. Too many, and large chunks
     of the signal could be getting smoothed/interpolated flat.
  3. Accel magnitude sanity check. MM-Fit's sp_r is gravity-included,
     so resting magnitude should hover close to 9.8 m/s^2. If your
     capture is way off (near 0, or way above ~15), that alone could
     explain windows reading as "non_activity" even during real
     movement - the model's non_activity class is essentially defined
     by "magnitude close to gravity, low variance."
  4. Gyro magnitude sanity check - flags a sensor units mismatch
     (rad/s vs deg/s), which would also make real reps look artificially
     calm to the model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_UPLOAD_DIR = PROJECT_ROOT / "data" / "sample_upload"

COLUMNS = ["frame", "timestamp_ms", "x", "y", "z"]


def load_raw(session_name: str, kind: str) -> pd.DataFrame:
    path = SAMPLE_UPLOAD_DIR / f"{session_name}_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def report_timing(df: pd.DataFrame, label: str):
    ts_s = df["timestamp_ms"].to_numpy() / 1000.0
    dt = np.diff(ts_s)
    dt = dt[dt > 0]  # guard against any zero/negative dt from bad timestamps

    sample_rate_hz = 1.0 / np.mean(dt)
    jitter_std_ms = np.std(dt) * 1000.0
    large_gaps = dt[dt > 0.1]  # >100ms treated as a "large" gap

    print(f"\n--- {label} timing ---")
    print(f"  samples:            {len(df)}")
    print(f"  duration:           {ts_s[-1] - ts_s[0]:.2f} s")
    print(f"  effective rate:     {sample_rate_hz:.1f} Hz")
    print(f"  jitter (std of dt): {jitter_std_ms:.2f} ms")
    print(f"  large gaps (>100ms): {len(large_gaps)}"
          f"{'  <-- ' + str(round(large_gaps.sum(), 2)) + 's total lost to gaps' if len(large_gaps) else ''}")


def report_magnitude(df: pd.DataFrame, label: str, expected_note: str):
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    print(f"\n--- {label} magnitude ---")
    print(f"  mean: {mag.mean():.3f}   std: {mag.std():.3f}   "
          f"min: {mag.min():.3f}   max: {mag.max():.3f}")
    print(f"  ({expected_note})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_live_raw.py <session_name>")
        sys.exit(1)

    session_name = sys.argv[1]
    acc_df = load_raw(session_name, "acc")
    gyro_df = load_raw(session_name, "gyro")

    print(f"Inspecting session: {session_name}")

    report_timing(acc_df, "Accel")
    report_timing(gyro_df, "Gyro")

    report_magnitude(
        acc_df, "Accel",
        "expected ~9.8 near rest if gravity-included (MM-Fit sp_r convention); "
        "near 0 would mean gravity-excluded data, which would look like a "
        "silent failure similar to using Accelerometer.csv instead of "
        "TotalAcceleration.csv"
    )
    report_magnitude(
        gyro_df, "Gyro",
        "expected small values (a few rad/s at most during real reps) if "
        "units are rad/s; if values are in the tens/hundreds, DeviceMotion "
        "may be reporting deg/s instead of rad/s, which would badly distort "
        "every gyro-derived feature and the rep counter (which runs on raw "
        "gyro magnitude)"
    )


if __name__ == "__main__":
    main()
