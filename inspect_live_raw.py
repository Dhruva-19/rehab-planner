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
    run_model_predictions(session_name)

def run_model_predictions(session_name: str):
    """
    Step 2: run the SAME predict_from_raw_csv() pipeline used by
    /ingest, but surface top-3 class probabilities per window instead
    of just the final predicted label. This is the key Phase B check:

      - if the real 5/8/9 cluster (tricep_extensions,
        dumbbell_shoulder_press, lateral_shoulder_raises) shows up in
        top-3 with bicep_curls only narrowly ahead -> likely
        drift/orientation.
      - if bicep_curls wins outright and 5/8/9 barely appear ->
        likely a pipeline/feature bug, not a model weakness.
    """
    sys.path.append(str(PROJECT_ROOT / "src" / "inference"))
    sys.path.append(str(PROJECT_ROOT / "src" / "preprocessing"))
    sys.path.append(str(PROJECT_ROOT / "src" / "feature_engineering"))

    from predict_pipeline import (
        load_model_bundle, process_uploaded_session,
        extract_window_features, MODEL_PATH, WINDOWS_NPZ_PATH,
    )
    from sliding_windows import slide_windows_over_chunk

    acc_csv = SAMPLE_UPLOAD_DIR / f"{session_name}_acc.csv"
    gyro_csv = SAMPLE_UPLOAD_DIR / f"{session_name}_gyro.csv"

    bundle = load_model_bundle(MODEL_PATH, WINDOWS_NPZ_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    label_encoder = bundle["label_encoder"]
    label_names = bundle["label_names"]

    chunks = process_uploaded_session(str(acc_csv), str(gyro_csv))
    all_windows = []
    for chunk in chunks:
        all_windows.extend(slide_windows_over_chunk(chunk))

    if not all_windows:
        print("\n[!] No windows produced -- capture may be too short.")
        return

    feature_rows, meta_rows = [], []
    for w in all_windows:
        feature_rows.append(extract_window_features(w["data"]))
        meta_rows.append({"start_time": w["start_time"]})

    features_df = pd.DataFrame(feature_rows).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)

    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        print(f"\n[BUG FOUND] Feature extraction is missing columns the "
              f"model expects: {missing_cols}")
        return

    features_df = features_df[feature_cols]

    pred_proba = model.predict_proba(features_df)
    encoded_classes = model.classes_

    print(f"\n--- Top-3 class probabilities per window "
          f"({len(all_windows)} windows) ---\n")

    cluster_names = {"tricep_extensions", "dumbbell_shoulder_press",
                      "lateral_shoulder_raises"}
    cluster_in_top3 = 0
    bicep_curls_top1 = 0

    for i in range(len(all_windows)):
        proba_row = pred_proba[i]
        top3_idx = np.argsort(proba_row)[::-1][:3]
        top3_int = label_encoder.inverse_transform(encoded_classes[top3_idx])
        top3_names = [label_names[lbl] for lbl in top3_int]
        top3_probs = proba_row[top3_idx]

        top3_str = ", ".join(f"{n}={p:.2f}" for n, p in zip(top3_names, top3_probs))
        print(f"  t={meta_rows[i]['start_time']:6.2f}s  ->  {top3_str}")

        if cluster_names & set(top3_names):
            cluster_in_top3 += 1
        if top3_names[0] == "bicep_curls":
            bicep_curls_top1 += 1

    print("\n--- Interpretation ---")
    print(f"  5/8/9 cluster present in top-3: {cluster_in_top3}/{len(all_windows)}")
    print(f"  bicep_curls is #1 prediction:   {bicep_curls_top1}/{len(all_windows)}")

    if cluster_in_top3 >= len(all_windows) * 0.5:
        print("\n  => 5/8/9 cluster IS showing up in top-3 for most windows. "
              "Points toward drift/orientation -- model is 'in the right "
              "neighborhood', bicep_curls just narrowly edges it out.")
    else:
        print("\n  => 5/8/9 cluster is largely ABSENT from top-3, bicep_curls "
              "wins outright. Points toward a pipeline/feature bug -- "
              "feature_cols reindexing, resampling, or windowing specific "
              "to the /ingest path -- not a model weakness.")
        
if __name__ == "__main__":
    main()
