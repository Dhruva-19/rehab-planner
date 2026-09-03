"""
src/inference/export_sample_upload.py

Purpose: Create a realistic "raw upload" test fixture by exporting one
MM-Fit session's raw sp_r_acc / sp_r_gyr .npy files to CSV, in the exact
[frame, timestamp_ms, x, y, z] format predict_pipeline.py expects from a
live upload.

This is a TEST-DATA GENERATOR only — it does not touch training data or
the model. Run it once, then point predict_pipeline.py at the CSVs it
produces to sanity-check the full inference pipeline end-to-end.

Usage:
    python src/inference/export_sample_upload.py w05
    (defaults to w05 if no session given — pick any session you know the
    dominant exercise label for, so you can eyeball whether the
    prediction makes sense)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("data/raw/mm-fit")
OUTPUT_DIR = Path("data/sample_upload")


def export_modality_to_csv(session: str, modality: str, out_path: Path):
    """
    Load one raw modality file (e.g. sp_r_acc) and write it out as CSV
    with columns [frame, timestamp_ms, x, y, z] -- matching the raw
    MM-Fit .npy layout exactly, so this is a genuine "unprocessed" file,
    not a pre-resampled one.
    """
    src_path = DATA_ROOT / session / f"{session}_{modality}.npy"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Could not find {src_path}. Check that '{session}' is a "
            f"valid session id and that raw data is downloaded."
        )

    arr = np.load(src_path)
    if arr.shape[1] < 5:
        raise ValueError(
            f"{src_path} has {arr.shape[1]} columns, expected >= 5 "
            f"[frame, timestamp_ms, x, y, z]."
        )

    df = pd.DataFrame(arr[:, :5], columns=["frame", "timestamp_ms", "x", "y", "z"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Exported {len(df)} rows -> {out_path}")


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "w05"

    print(f"Exporting raw acc + gyro CSVs for session '{session}'...")
    export_modality_to_csv(session, "sp_r_acc", OUTPUT_DIR / "session_acc.csv")
    export_modality_to_csv(session, "sp_r_gyr", OUTPUT_DIR / "session_gyro.csv")

    print(
        f"\nDone. Test the pipeline with:\n"
        f'  ACC_CSV  = "data/sample_upload/session_acc.csv"\n'
        f'  GYRO_CSV = "data/sample_upload/session_gyro.csv"\n'
        f"\nTip: check MM-Fit's labels.csv for session '{session}' first "
        f"to know which exercise(s) this session actually contains, so "
        f"you can compare against the pipeline's prediction."
    )


if __name__ == "__main__":
    main()
