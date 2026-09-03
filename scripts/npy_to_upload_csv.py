"""
scripts/npy_to_upload_csv.py

Purpose: Convert a MM-Fit session's raw sp_r_acc.npy / sp_r_gyr.npy files
into CSVs matching the schema predict_from_raw_csv() expects:
    [frame, timestamp_ms, x, y, z]   (no header)

This simulates what a real phone sensor-logging app would export, so we
can test the New Session upload tab end-to-end using a held-out MM-Fit
session as stand-in "live" data.

Usage:
    python scripts/npy_to_upload_csv.py w00
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/mm-fit")
OUTPUT_DIR = Path("data/sample_upload")

EXPECTED_COLUMNS = ["frame", "timestamp_ms", "x", "y", "z"]


def convert_npy_to_csv(npy_path: Path, csv_path: Path) -> None:
    """Load one raw sensor .npy file and write it out as a 5-column CSV."""
    if not npy_path.exists():
        raise FileNotFoundError(f"Expected raw file not found: {npy_path}")

    arr = np.load(npy_path)

    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            f"{npy_path} has shape {arr.shape}, expected (N, 5) matching "
            f"{EXPECTED_COLUMNS}. Inspect the file before proceeding."
        )

    df = pd.DataFrame(arr, columns=EXPECTED_COLUMNS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # No header, no index — matches what read_raw_sensor_csv() auto-detects
    # as headerless numeric data (mirrors a real raw sensor export).
    df.to_csv(csv_path, header=False, index=False)

    print(f"  Wrote {len(df):,} rows -> {csv_path}")


def convert_session(session_id: str) -> None:
    """Convert both sp_r_acc and sp_r_gyr for one MM-Fit session."""
    session_dir = RAW_DIR / session_id

    acc_npy = session_dir / f"{session_id}_sp_r_acc.npy"
    gyr_npy = session_dir / f"{session_id}_sp_r_gyr.npy"

    acc_csv = OUTPUT_DIR / f"{session_id}_sp_r_acc.csv"
    gyr_csv = OUTPUT_DIR / f"{session_id}_sp_r_gyr.csv"

    print(f"Converting session '{session_id}':")
    convert_npy_to_csv(acc_npy, acc_csv)
    convert_npy_to_csv(gyr_npy, gyr_csv)
    print(f"\nDone. Upload these two files in the New Session tab:")
    print(f"  Accelerometer: {acc_csv}")
    print(f"  Gyroscope:     {gyr_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/npy_to_upload_csv.py <session_id>")
        print("Example: python scripts/npy_to_upload_csv.py w00")
        sys.exit(1)

    convert_session(sys.argv[1])
