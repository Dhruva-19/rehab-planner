"""
src/inference/predict_pipeline.py

Purpose: End-to-end inference on a freshly uploaded, raw (unresampled)
accelerometer + gyroscope CSV pair. Mirrors the exact preprocessing,
windowing, and feature-extraction logic used at training time, so a
live prediction is computed identically to how the training data was
built.

Pipeline:
    raw CSV (acc, gyro)
      -> resample to 200Hz + gap-splitting   (resample_pipeline.py, reused)
      -> 600-sample / 300-step sliding windows (sliding_windows.py, reused)
      -> 113 features per window              (extract_features.py +
                                                 extract_advanced_features.py, reused)
      -> XGBoost prediction + confidence
"""

import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# -----------------------------
# Make sibling pipeline modules importable
# (adjust these two lines if your folder names differ)
# -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_PROJECT_ROOT / "preprocessing"))
sys.path.append(str(_PROJECT_ROOT / "feature_engineering"))

from resample_pipeline import (
    split_at_gaps,
    resample_segment,
    TARGET_HZ,
    GAP_THRESHOLD_S,
    MIN_CHUNK_DURATION_S,
)
from sliding_windows import slide_windows_over_chunk, WINDOW_SIZE
from extract_features import extract_features_from_window
from extract_advanced_features import extract_advanced_features_single_window

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "saved_models/xgboost_v2_orientation_features.pkl"
WINDOWS_NPZ_PATH = "data/processed/labeled_windows.npz"  # only read for label_names


# -----------------------------
# Model loading
# -----------------------------
def load_model_bundle(model_path: str = MODEL_PATH,
                       windows_npz_path: str = WINDOWS_NPZ_PATH) -> dict:
    """
    Load the trained model + feature column order + label encoder, and
    attach label_names (int -> human-readable class string) from the
    original windows file so predictions can be decoded to text.
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    data = np.load(windows_npz_path, allow_pickle=True)
    bundle["label_names"] = data["label_names"]

    required_keys = {"model", "feature_cols", "label_encoder", "label_names"}
    missing = required_keys - bundle.keys()
    if missing:
        raise ValueError(f"Model bundle is missing expected keys: {missing}")

    return bundle


# -----------------------------
# Raw CSV reading (replaces load_and_normalize, which reads .npy by session)
# -----------------------------
def read_raw_sensor_csv(csv_path: str):
    """
    Read one raw sensor CSV.

    Expected columns, in order: [frame, timestamp_ms, x, y, z]
    Header row is optional and auto-detected.

    Returns:
        timestamps: (T,) array, in SECONDS (matches training convention)
        xyz:        (T, 3) array
    """
    df = pd.read_csv(csv_path, header=None, low_memory=False)

    # If the first cell of the first row isn't numeric, assume it's a
    # header row and re-read letting pandas handle it, then drop labels.
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = range(df.shape[1])  # normalize back to positional columns

    df = df.apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [{csv_path}] dropped {n_dropped} malformed row(s)")

    if df.shape[1] < 5:
        raise ValueError(
            f"Expected >= 5 columns [frame, timestamp_ms, x, y, z] in "
            f"{csv_path}, found {df.shape[1]}."
        )
    if len(df) < 2:
        raise ValueError(f"{csv_path} has too few valid rows to process.")

    arr = df.to_numpy(dtype=np.float64)
    timestamps = arr[:, 1] / 1000.0   # ms -> seconds, same as training
    xyz = arr[:, 2:5]
    return timestamps, xyz


# -----------------------------
# Resample + chunk (mirrors resample_pipeline.process_session)
# -----------------------------
def process_uploaded_session(acc_csv_path: str, gyro_csv_path: str) -> list:
    """
    Same logic as resample_pipeline.process_session(), but sourced from
    two uploaded CSVs instead of a fixed data/raw/mm-fit/{session}/ path.
    """
    acc_ts, acc_xyz = read_raw_sensor_csv(acc_csv_path)
    gyr_ts, gyr_xyz = read_raw_sensor_csv(gyro_csv_path)

    acc_segments = split_at_gaps(acc_ts, acc_xyz, GAP_THRESHOLD_S)

    chunks = []
    for seg_ts, seg_xyz in acc_segments:
        duration = seg_ts[-1] - seg_ts[0]
        if duration < MIN_CHUNK_DURATION_S:
            continue

        uniform_ts, acc_resampled = resample_segment(seg_ts, seg_xyz, TARGET_HZ)
        if uniform_ts is None:
            continue

        # Interpolate gyro onto the SAME uniform timeline as acc
        gyr_resampled = np.column_stack([
            np.interp(uniform_ts, gyr_ts, gyr_xyz[:, axis]) for axis in range(3)
        ])

        chunks.append({
            "start_time": uniform_ts[0],
            "timestamps": uniform_ts,
            "acc": acc_resampled,
            "gyr": gyr_resampled,
        })

    if not chunks:
        raise ValueError(
            "No usable chunks produced from the uploaded CSVs. Check that "
            "the recording is long enough (>= "
            f"{MIN_CHUNK_DURATION_S}s continuous) and timestamps look sane."
        )

    return chunks


# -----------------------------
# Feature extraction per window (merges base + advanced, in that order)
# -----------------------------
def extract_window_features(window: np.ndarray) -> dict:
    """window: (600, 6) -> merged 113-feature dict."""
    base_feats = extract_features_from_window(window)
    advanced_feats = extract_advanced_features_single_window(window)
    return {**base_feats, **advanced_feats}


# -----------------------------
# Full pipeline: raw CSV -> prediction
# -----------------------------
def predict_from_raw_csv(acc_csv_path: str,
                          gyro_csv_path: str,
                          bundle: dict = None,
                          model_path: str = MODEL_PATH,
                          windows_npz_path: str = WINDOWS_NPZ_PATH) -> pd.DataFrame:
    """
    Runs the full pipeline on one uploaded acc/gyro CSV pair.

    Returns a DataFrame, one row per predicted window:
        start_time, end_time, predicted_label (int),
        predicted_class_name (str), confidence (float, 0-1)
    """
    if bundle is None:
        bundle = load_model_bundle(model_path, windows_npz_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    label_encoder = bundle["label_encoder"]
    label_names = bundle["label_names"]

    chunks = process_uploaded_session(acc_csv_path, gyro_csv_path)

    all_windows = []
    for chunk in chunks:
        all_windows.extend(slide_windows_over_chunk(chunk))

    if not all_windows:
        raise ValueError(
            f"No {WINDOW_SIZE}-sample window could be formed from the "
            f"uploaded data. Need at least {WINDOW_SIZE / TARGET_HZ:.1f}s "
            f"of continuous, resampled data."
        )

    feature_rows, meta_rows = [], []
    for w in all_windows:
        feature_rows.append(extract_window_features(w["data"]))
        meta_rows.append({"start_time": w["start_time"], "end_time": w["end_time"]})

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Feature extraction is missing expected columns: {missing_cols}")

    # Enforce the EXACT column order the model was trained on.
    features_df = features_df[feature_cols]

    pred_encoded = model.predict(features_df)
    pred_proba = model.predict_proba(features_df)
    confidence = pred_proba.max(axis=1)

    pred_int_labels = label_encoder.inverse_transform(pred_encoded)
    pred_class_names = [label_names[i] for i in pred_int_labels]

    results = pd.DataFrame(meta_rows)
    results["predicted_label"] = pred_int_labels
    results["predicted_class_name"] = pred_class_names
    results["confidence"] = confidence

    return results


if __name__ == "__main__":
    # Quick manual test — point these at a sample acc/gyro CSV pair.
    ACC_CSV = "data/sample_upload/session_acc.csv"
    GYRO_CSV = "data/sample_upload/session_gyro.csv"

    results = predict_from_raw_csv(ACC_CSV, GYRO_CSV)

    print(f"\nProduced {len(results)} window predictions:\n")
    print(results.to_string(index=False))

    print("\nMajority vote across the uploaded recording:")
    print(results["predicted_class_name"].value_counts())