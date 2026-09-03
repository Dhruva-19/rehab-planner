"""
Purpose: Convert Sensor Logger app's exported TotalAcceleration.csv /
Gyroscope.csv into the schema predict_from_raw_csv() expects:
    [frame, timestamp_ms, x, y, z]   (no header)

Sensor Logger's own export columns are different from MM-Fit's raw .npy
layout:
    time            -> nanoseconds since Unix epoch
    seconds_elapsed -> seconds since "Start Recording" was tapped
    x, y, z         -> sensor readings

IMPORTANT: use TotalAcceleration.csv, NOT Accelerometer.csv, for the
accelerometer stream. Verified against MM-Fit's real sp_r_acc.npy (w00):
mean acceleration magnitude ~9.97, confirming MM-Fit's raw accelerometer
INCLUDES gravity. Sensor Logger's Accelerometer.csv excludes gravity
(near-zero magnitude at rest), which caused every set to be misclassified
as non_activity with high confidence during Day 11 testing — the model
had never seen a signal that flat during training. Gyroscope.csv is
unaffected by this issue and stays the same either way.
We use `time` (converted ns -> ms) as timestamp_ms, since that's what the
resample pipeline expects (an absolute, monotonically increasing clock),
matching how MM-Fit's own timestamp_ms column works.

`frame` is not read anywhere downstream (read_raw_sensor_csv only uses
columns 1-4), so it's safely filled with a placeholder of 0 for real
phone recordings, which have no associated video to align to.

Usage:
    python scripts/sensorlogger_to_upload_csv.py \\
        path/to/TotalAcceleration.csv path/to/Gyroscope.csv my_session_name"""

import sys
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/sample_upload")


def convert_sensorlogger_csv(src_csv: Path, dst_csv: Path) -> None:
    """Read one Sensor Logger export CSV and write it in the target schema."""
    if not src_csv.exists():
        raise FileNotFoundError(f"Expected Sensor Logger export not found: {src_csv}")

    df = pd.read_csv(src_csv)

    required = {"time", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{src_csv} is missing expected Sensor Logger column(s): {missing}. "
            f"Found columns: {list(df.columns)}. Make sure this is a raw "
            f"Accelerometer.csv or Gyroscope.csv from the Sensor Logger export zip."
        )

    out = pd.DataFrame({
        "frame": 0,                          # unused downstream, placeholder
        "timestamp_ms": df["time"] / 1e6,    # ns -> ms
        "x": df["x"],
        "y": df["y"],
        "z": df["z"],
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_csv, header=False, index=False)
    print(f"  Wrote {len(out):,} rows -> {dst_csv}")


def convert_session(acc_src: str, gyro_src: str, session_name: str) -> None:
    acc_csv = OUTPUT_DIR / f"{session_name}_acc.csv"
    gyro_csv = OUTPUT_DIR / f"{session_name}_gyro.csv"

    print(f"Converting Sensor Logger recording '{session_name}':")
    convert_sensorlogger_csv(Path(acc_src), acc_csv)
    convert_sensorlogger_csv(Path(gyro_src), gyro_csv)
    print(f"\nDone. Upload these two files in the New Session tab:")
    print(f"  Accelerometer: {acc_csv}")
    print(f"  Gyroscope:     {gyro_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/sensorlogger_to_upload_csv.py <Accelerometer.csv> <Gyroscope.csv> <session_name>")
        print("Example: python scripts/sensorlogger_to_upload_csv.py Accelerometer.csv Gyroscope.csv my_squats")
        sys.exit(1)

    convert_session(sys.argv[1], sys.argv[2], sys.argv[3])
