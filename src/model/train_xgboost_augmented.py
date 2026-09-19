"""
src/model/train_xgboost_augmented.py

Day 25 - Step: retrain XGBoost on MM-Fit + live-captured data (augmented).

Reuses the exact same hyperparameters (XGB_PARAMS) as the original
train_xgboost.py, imported directly (not retyped) - so this retrain is a
fair, apples-to-apples comparison against the current production model:
only the training DATA changes (MM-Fit + 114 live-captured windows across
8 exercise classes), not the model architecture or hyperparameters.

Versioning (Day 25 decision - see tools-and-resources.md): saves under a
NEW filename, xgboost_v4_augmented.pkl. Never overwrites the existing
xgboost_v2_orientation_features.pkl in place. predict_pipeline.py's
MODEL_PATH is NOT changed by this script - that only happens manually,
after validating this new model doesn't regress on the original held-out
MM-Fit test set (via evaluate_held_out.py).

Run from your project root, same as train_xgboost.py.
"""

import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from train_xgboost import XGB_PARAMS, TRAIN_VAL_PATH  # reuse exact same hyperparameters

LIVE_FEATURES_PATH = "data/processed/live_features.csv"
NEW_MODEL_PATH = Path("saved_models/xgboost_v4_augmented.pkl")


def load_and_merge() -> pd.DataFrame:
    """Load the original MM-Fit train_val set + the new live-captured features."""
    df_mmfit = pd.read_csv(TRAIN_VAL_PATH)
    df_live = pd.read_csv(LIVE_FEATURES_PATH)

    # Sanity check: the feature columns must match exactly (aside from
    # label/session), or something diverged in the reused pipeline chain -
    # fail loudly here rather than silently training on misaligned columns.
    mmfit_cols = set(df_mmfit.columns) - {"label", "session"}
    live_cols = set(df_live.columns) - {"label", "session"}
    if mmfit_cols != live_cols:
        missing_in_live = mmfit_cols - live_cols
        missing_in_mmfit = live_cols - mmfit_cols
        raise ValueError(
            "Feature column mismatch between train_val.csv and live_features.csv.\n"
            f"In train_val.csv but not live_features.csv: {missing_in_live}\n"
            f"In live_features.csv but not train_val.csv: {missing_in_mmfit}"
        )

    print(f"MM-Fit rows: {len(df_mmfit)}, live rows: {len(df_live)}")
    df_combined = pd.concat([df_mmfit, df_live], ignore_index=True)
    print(f"Combined training set: {len(df_combined)} rows")
    return df_combined


def main():
    df = load_and_merge()
    feature_cols = [c for c in df.columns if c not in ("label", "session")]

    # Label encoder fit on the COMBINED labels - MM-Fit's labels are already
    # the ACTIONS integer indices, and live_features.csv uses the identical
    # encoding (see label_live_captures.py), so this is just re-confirming
    # the same 0-10 mapping, consistent with train_xgboost.py's own approach.
    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    print(f"Label classes: {list(label_encoder.classes_)}")

    X = df[feature_cols]
    y = label_encoder.transform(df["label"])

    print("\nTraining XGBoost on the full augmented dataset (one fit() call)...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    NEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "label_encoder": label_encoder,
        }, f)

    print(f"\nSaved augmented model -> {NEW_MODEL_PATH}")
    print("NOTE: predict_pipeline.py's MODEL_PATH has NOT been changed. "
          "Validate against the held-out MM-Fit test set (evaluate_held_out.py) "
          "before flipping MODEL_PATH to this new file.")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 15 most important features:")
    print(importances.sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
