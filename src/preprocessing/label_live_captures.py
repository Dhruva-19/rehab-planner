"""
src/preprocessing/label_live_captures.py

Day 25 - Step: Label live-captured exercise data for retraining.

Goal: turn the 8 usable live-captured exercise sessions (phone-in-pocket,
browser DeviceMotion, ~55-60Hz) into rows in the EXACT same format as
data/processed/train_val.csv, so they can be concatenated straight in
before retraining.

Design decisions (confirmed with Dhruva, Day 25):
  - Reuse the training-time pipeline via predict_pipeline.py's own
    functions (process_uploaded_session, slide_windows_over_chunk,
    extract_window_features) rather than reimplementing resampling or
    feature extraction, to guarantee identical feature space to
    train_val.csv. extract_window_features() = base + advanced features
    only (113 cols) - matches what the DEPLOYED model actually uses
    (confirmed: predict_pipeline.py never calls the orientation feature
    extractor, despite the model's filename).
  - Segmentation (where are the real reps vs. rest, including mid-file
    rest gaps in multi-set files like Biceps_curls) is done by running
    the EXISTING model + aggregate_into_sets() and using only the
    resulting time boundaries. The model's own class guess for each set
    is discarded entirely and replaced with our own known ground-truth
    label from the filename - we don't trust its classification (that's
    what we're fixing), only its active-vs-rest segmentation.
  - non_activity windows are NOT included in this output at all.
    MM-Fit already supplies ample non_activity data (already downsampled
    to 2.5x the largest exercise class in build_labeled_dataset.py);
    adding more would risk re-imbalancing for zero benefit, since the
    diagnosed problems are confusion BETWEEN exercise classes, not
    rest-vs-active detection.
  - triceps extensions and lateral raises are excluded (documented
    limitation - live captures of these two didn't classify reliably;
    see learnings-and-limitations.md).
  - Lunges was captured across 3 short sessions (2+4+4 reps) due to
    indoor space constraints; Jumping Jacks across 2. Each capture file
    is treated as its own "session" for the output's session column,
    same convention as train_val.csv's MM-Fit sessions (w00, w01, ...).

Run from your project root (same folder as train_xgboost.py's sibling
dirs are importable from).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Make sibling pipeline modules importable (same pattern as predict_pipeline.py)
# -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_PROJECT_ROOT / "preprocessing"))
sys.path.append(str(_PROJECT_ROOT / "feature_engineering"))
sys.path.append(str(_PROJECT_ROOT / "inference"))
sys.path.append(str(_PROJECT_ROOT / "feedback"))

from predict_pipeline import (
    process_uploaded_session,
    extract_window_features,
    predict_from_raw_csv,
    load_model_bundle,
)
from sliding_windows import slide_windows_over_chunk
from aggregate_sets import aggregate_into_sets

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data/sample_upload")
OUTPUT_PATH = Path("data/processed/live_features.csv")

# Same class-index order as label_alignment.py's ACTIONS - MUST match exactly,
# since train_val.csv's "label" column is this integer index, not a string.
ACTIONS = [
    "squats", "lunges", "bicep_curls", "situps", "pushups",
    "tricep_extensions", "dumbbell_rows", "jumping_jacks",
    "dumbbell_shoulder_press", "lateral_shoulder_raises", "non_activity",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(ACTIONS)}

# filename base (without _acc.csv / _gyro.csv) -> ground-truth class name.
# Confirmed against live app output, not guessed from filename spelling.
FILENAME_TO_CLASS = {
    "Biceps_curls":            "bicep_curls",
    "Dumbbell_shoulder_press": "dumbbell_shoulder_press",
    "Jumping_jacks":           "jumping_jacks",
    "Jumping_jacks2":          "jumping_jacks",
    "Lunges":                  "lunges",
    "Lunges3":                 "lunges",
    "Lunges4":                 "lunges",
    "Pushups":                 "pushups",
    "Situps":                  "situps",
    "Squats":                  "squats",
    "Standing_dumbell_rows":  "dumbbell_rows",
}


def find_active_time_ranges(acc_csv: str, gyro_csv: str, bundle: dict, min_run: int = 2) -> list:
    """
    Find active-vs-rest time boundaries directly from RAW per-window
    predictions (not the smoothed/aggregated sets). aggregate_into_sets()'s
    5-window mode-smoothing is tuned for MM-Fit's dense continuous data and
    erases short real bursts in live rehab-paced captures (confirmed via
    the Lunges diagnostic - see learnings-and-limitations.md). A run of
    >= min_run consecutive non-"non_activity" windows counts as active;
    isolated single-window blips (sensor noise) are dropped.
    """
    window_preds = predict_from_raw_csv(acc_csv, gyro_csv, bundle=bundle)
    window_preds = window_preds.sort_values("start_time").reset_index(drop=True)
    is_active = (window_preds["predicted_class_name"] != "non_activity").to_numpy()

    ranges = []
    i, n = 0, len(is_active)
    while i < n:
        if is_active[i]:
            j = i
            while j < n and is_active[j]:
                j += 1
            if (j - i) >= min_run:
                start = window_preds.loc[i, "start_time"]
                end = window_preds.loc[j - 1, "end_time"]
                ranges.append((start, end))
            i = j
        else:
            i += 1
    return ranges

def label_one_capture(base_name: str, true_class: str, bundle: dict) -> list:
    """
    Process one capture file pair -> list of (feature_dict, label_idx, session)
    rows, covering only the windows that fall inside an active time range.
    """
    acc_csv = str(DATA_DIR / f"{base_name}_acc.csv")
    gyro_csv = str(DATA_DIR / f"{base_name}_gyro.csv")

    active_ranges = find_active_time_ranges(acc_csv, gyro_csv, bundle)
    if not active_ranges:
        print(f"  WARNING: {base_name} - no active segments detected, skipping.")
        return []

    chunks = process_uploaded_session(acc_csv, gyro_csv)
    all_windows = []
    for chunk in chunks:
        all_windows.extend(slide_windows_over_chunk(chunk))

    label_idx = LABEL_TO_IDX[true_class]
    session_name = f"live_{base_name}"

    rows = []
    for w in all_windows:
        w_mid = (w["start_time"] + w["end_time"]) / 2.0
        in_active_segment = any(start <= w_mid <= end for start, end in active_ranges)
        if not in_active_segment:
            continue
        feats = extract_window_features(w["data"])
        feats["label"] = label_idx
        feats["session"] = session_name
        rows.append(feats)

    print(f"  {base_name:<28} -> {len(all_windows):>4} total windows, "
          f"{len(rows):>4} kept as '{true_class}' "
          f"({len(active_ranges)} active segment(s) detected)")
    return rows


def main():
    print("Loading model bundle (used only for active/rest segmentation)...")
    bundle = load_model_bundle()

    print(f"\nProcessing {len(FILENAME_TO_CLASS)} live capture files...\n")
    all_rows = []
    for base_name, true_class in FILENAME_TO_CLASS.items():
        rows = label_one_capture(base_name, true_class, bundle)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No live windows were labeled - check data/sample_upload/ paths.")

    df = pd.DataFrame(all_rows)

    print(f"\nTotal live windows labeled: {len(df)}")
    print("\nClass distribution (live data only):")
    for idx, count in df["label"].value_counts().sort_index().items():
        print(f"  {ACTIONS[idx]:<28} {count:>4}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")
    print("Next step: concatenate this with train_val.csv and retrain "
          "(saving under a NEW model filename, per the Day 25 versioning plan).")


if __name__ == "__main__":
    main()
