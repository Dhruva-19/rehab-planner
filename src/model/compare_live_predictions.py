"""
src/model/compare_live_predictions.py

Day 25 - Final step: compare the OLD (v2, MM-Fit only) vs NEW (v4,
augmented) model on the actual live capture files. This is the real
test of whether today's retrain helped, since evaluate_held_out.py's
MM-Fit-only test set can't measure live-data performance at all.

For each live capture file, both models run through the exact same
predict_from_raw_csv() pipeline. Since we already know the TRUE class
for every file (label_live_captures.py's confirmed FILENAME_TO_CLASS
mapping, imported directly here so it can't drift out of sync), we
measure per window:
  - correct: predicted the true exercise
  - wrong_exercise: predicted a DIFFERENT exercise
  - non_activity: predicted rest

"among-active accuracy" = correct / (total - non_activity) isolates
exercise-classification quality specifically, independent of
segmentation, since segmentation wasn't what Day 25 targeted.

Run from your project root, same as the other src/model/ scripts.
"""

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_PROJECT_ROOT / "preprocessing"))
sys.path.append(str(_PROJECT_ROOT / "feature_engineering"))
sys.path.append(str(_PROJECT_ROOT / "inference"))
sys.path.append(str(_PROJECT_ROOT / "feedback"))

from predict_pipeline import predict_from_raw_csv, load_model_bundle
from label_live_captures import FILENAME_TO_CLASS  # reuse the confirmed mapping

DATA_DIR = Path("data/sample_upload")
OLD_MODEL_PATH = "saved_models/xgboost_v2_orientation_features.pkl"
NEW_MODEL_PATH = "saved_models/xgboost_v4_augmented.pkl"


def score_file(base_name: str, true_class: str, bundle: dict) -> dict:
    acc_csv = str(DATA_DIR / f"{base_name}_acc.csv")
    gyro_csv = str(DATA_DIR / f"{base_name}_gyro.csv")

    preds = predict_from_raw_csv(acc_csv, gyro_csv, bundle=bundle)
    total = len(preds)

    is_non_activity = preds["predicted_class_name"] == "non_activity"
    is_correct = preds["predicted_class_name"] == true_class
    is_wrong_exercise = (~is_non_activity) & (~is_correct)

    n_non_activity = int(is_non_activity.sum())
    n_correct = int(is_correct.sum())
    n_wrong = int(is_wrong_exercise.sum())
    n_active = total - n_non_activity

    among_active_acc = n_correct / n_active if n_active > 0 else float("nan")
    mean_conf_correct = (
        preds.loc[is_correct, "confidence"].mean() if n_correct > 0 else float("nan")
    )

    return {
        "total": total,
        "correct": n_correct,
        "wrong_exercise": n_wrong,
        "non_activity": n_non_activity,
        "among_active_acc": among_active_acc,
        "mean_conf_correct": mean_conf_correct,
    }


def main():
    print("Loading both model bundles...")
    old_bundle = load_model_bundle(model_path=OLD_MODEL_PATH)
    new_bundle = load_model_bundle(model_path=NEW_MODEL_PATH)

    rows = []
    for base_name, true_class in FILENAME_TO_CLASS.items():
        old_r = score_file(base_name, true_class, old_bundle)
        new_r = score_file(base_name, true_class, new_bundle)
        rows.append({
            "file": base_name,
            "true_class": true_class,
            "old_acc": old_r["among_active_acc"],
            "new_acc": new_r["among_active_acc"],
            "old_correct/total": f"{old_r['correct']}/{old_r['total']}",
            "new_correct/total": f"{new_r['correct']}/{new_r['total']}",
            "old_conf": old_r["mean_conf_correct"],
            "new_conf": new_r["mean_conf_correct"],
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n" + "=" * 100)
    print("LIVE CAPTURE COMPARISON: v2 (MM-Fit only) vs v4 (augmented)")
    print("=" * 100)
    print(df.to_string(index=False))

    valid = df.dropna(subset=["old_acc", "new_acc"])
    print("\nOverall mean 'among-active' accuracy across all live files:")
    print(f"  v2 (old): {valid['old_acc'].mean():.3f}")
    print(f"  v4 (new): {valid['new_acc'].mean():.3f}")

    print("\nOriginally-diagnosed problem exercises (squats, pushups, situps):")
    flagged = df[df["true_class"].isin(["squats", "pushups", "situps"])]
    print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
