"""
src/model/split_data.py

Purpose: Create a leakage-free train/test split at the SESSION level, then
set up cross-validation folds (also session-aware) for model comparison.

Why this matters: windows from the same session/person must never appear
in both train and test, or accuracy numbers become meaningless (data leakage).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# -----------------------------
# Config
# -----------------------------
FEATURES_PATH = "data/processed/features_v2.csv"
OUTPUT_DIR = Path("data/processed")
N_FOLDS = 5
HELD_OUT_TEST_FRACTION = 0.20   # ~4 out of 21 sessions
RANDOM_STATE = 42                # fixed seed = reproducible splits


def load_features(path: str) -> pd.DataFrame:
    """Load the feature CSV produced in Day 4."""
    df = pd.read_csv(path)
    print(f"Loaded features.csv: shape={df.shape}")
    print(f"Columns preview: {list(df.columns[:5])} ... {list(df.columns[-5:])}")

    required_cols = {"label", "session"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"features.csv is missing required column(s): {missing}. "
            f"Check extract_features.py output before proceeding."
        )
    return df


def carve_out_held_out_test(df: pd.DataFrame):
    """
    Split sessions into (train_val_sessions, held_out_test_sessions) using
    GroupShuffleSplit — this guarantees a full session goes to ONE side only.

    Returns two DataFrames: df_train_val, df_held_out_test
    """
    groups = df["session"]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=HELD_OUT_TEST_FRACTION,
        random_state=RANDOM_STATE
    )

    # GroupShuffleSplit gives us row INDICES, split so no group crosses over
    train_val_idx, held_out_idx = next(gss.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_held_out_test = df.iloc[held_out_idx].reset_index(drop=True)

    return df_train_val, df_held_out_test


def build_cv_folds(df_train_val: pd.DataFrame):
    """
    Build StratifiedGroupKFold indices on the train_val portion only.
    Each fold trains on ~4/5 of sessions, validates on the held-back 1/5.

    Returns a list of (train_idx, val_idx) tuples (indices into df_train_val).
    """
    X_placeholder = df_train_val.drop(columns=["label", "session"])
    y = df_train_val["label"]
    groups = df_train_val["session"]

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(sgkf.split(X_placeholder, y, groups=groups))
    return folds


def verify_split_quality(df_train_val: pd.DataFrame,
                          df_held_out_test: pd.DataFrame,
                          folds: list):
    """
    Verifies:
      1. Zero session overlap between train_val and held_out_test.
      2. Class distribution is roughly similar across train_val vs held_out_test.
      3. For each CV fold: no session overlap between its train/val split,
         and no exercise class is completely missing from a val fold.
    """
    print("\n" + "=" * 60)
    print("1. HELD-OUT TEST vs TRAIN_VAL — SESSION OVERLAP CHECK")
    print("=" * 60)

    train_val_sessions = set(df_train_val["session"].unique())
    held_out_sessions = set(df_held_out_test["session"].unique())
    overlap = train_val_sessions & held_out_sessions

    print(f"Train_val sessions ({len(train_val_sessions)}): {sorted(train_val_sessions)}")
    print(f"Held-out test sessions ({len(held_out_sessions)}): {sorted(held_out_sessions)}")

    assert len(overlap) == 0, f"LEAKAGE DETECTED! Overlapping sessions: {overlap}"
    print("PASS: No session overlap between train_val and held_out_test.")

    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION — TRAIN_VAL vs HELD_OUT_TEST")
    print("=" * 60)

    train_val_dist = df_train_val["label"].value_counts(normalize=True).sort_index()
    held_out_dist = df_held_out_test["label"].value_counts(normalize=True).sort_index()

    comparison = pd.DataFrame({
        "train_val_%": (train_val_dist * 100).round(2),
        "held_out_%": (held_out_dist * 100).round(2)
    })
    comparison["abs_diff_%"] = (comparison["train_val_%"] - comparison["held_out_%"]).abs()
    print(comparison)

    max_diff = comparison["abs_diff_%"].max()
    if max_diff > 10:
        print(f"\nWARNING: Class '{comparison['abs_diff_%'].idxmax()}' differs by "
              f"{max_diff:.2f}% between splits. Consider trying a different "
              f"RANDOM_STATE if this looks too skewed.")
    else:
        print(f"\nOK: Max class distribution difference is {max_diff:.2f}% — acceptable.")

    print("\n" + "=" * 60)
    print("3. CROSS-VALIDATION FOLD CHECKS")
    print("=" * 60)

    all_labels = set(df_train_val["label"].unique())

    for i, (train_idx, val_idx) in enumerate(folds):
        train_sessions = set(df_train_val.iloc[train_idx]["session"])
        val_sessions = set(df_train_val.iloc[val_idx]["session"])

        # Structural leakage check (should be impossible with StratifiedGroupKFold,
        # but verify — trust, then verify)
        fold_overlap = train_sessions & val_sessions
        assert len(fold_overlap) == 0, (
            f"Fold {i}: LEAKAGE — sessions {fold_overlap} appear in both "
            f"train and val!"
        )

        val_labels_present = set(df_train_val.iloc[val_idx]["label"].unique())
        missing_labels = all_labels - val_labels_present

        print(f"\nFold {i}:")
        print(f"  Train sessions ({len(train_sessions)}): {sorted(train_sessions)}")
        print(f"  Val sessions   ({len(val_sessions)}): {sorted(val_sessions)}")
        print(f"  Train rows: {len(train_idx)}, Val rows: {len(val_idx)}")

        if missing_labels:
            print(f"  WARNING: classes missing from val fold: {missing_labels}")
        else:
            print(f"  OK: all {len(all_labels)} classes present in val fold.")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


def save_splits(df_train_val, df_held_out_test, folds):
    """
    Persist the split so training scripts can load it later without
    re-running the random split (keeps everything reproducible).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train_val.to_csv(OUTPUT_DIR / "train_val.csv", index=False)
    df_held_out_test.to_csv(OUTPUT_DIR / "held_out_test.csv", index=False)

    # Save fold indices as a JSON of lists (indices are into df_train_val)
    fold_dict = {
        f"fold_{i}": {"train_idx": tr.tolist(), "val_idx": va.tolist()}
        for i, (tr, va) in enumerate(folds)
    }
    with open(OUTPUT_DIR / "cv_folds.json", "w") as f:
        json.dump(fold_dict, f)

    print(f"\nSaved: train_val.csv ({len(df_train_val)} rows), "
          f"held_out_test.csv ({len(df_held_out_test)} rows), "
          f"cv_folds.json ({N_FOLDS} folds)")


if __name__ == "__main__":
    df = load_features(FEATURES_PATH)
    df_train_val, df_held_out_test = carve_out_held_out_test(df)
    folds = build_cv_folds(df_train_val)
    verify_split_quality(df_train_val, df_held_out_test, folds)
    save_splits(df_train_val, df_held_out_test, folds)