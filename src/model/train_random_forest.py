"""
src/model/train_random_forest.py

Purpose: Train a Random Forest baseline using session-aware CV folds,
report per-fold and pooled metrics, and save the final model trained
on all train_val data (for later comparison against XGBoost / LSTM).
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Config
# -----------------------------
TRAIN_VAL_PATH = "data/processed/train_val.csv"
FOLDS_PATH = "data/processed/cv_folds.json"
MODEL_OUTPUT_DIR = Path("saved_models")
RESULTS_OUTPUT_DIR = Path("data/processed")

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}


def load_data():
    """Load the train_val feature set and the pre-computed CV fold indices."""
    df = pd.read_csv(TRAIN_VAL_PATH)

    with open(FOLDS_PATH, "r") as f:
        fold_dict = json.load(f)

    # Reconstruct list of (train_idx, val_idx) tuples as numpy arrays
    folds = [
        (np.array(fold_dict[f"fold_{i}"]["train_idx"]),
         np.array(fold_dict[f"fold_{i}"]["val_idx"]))
        for i in range(len(fold_dict))
    ]

    feature_cols = [c for c in df.columns if c not in ("label", "session")]
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, {len(folds)} folds.")
    return df, feature_cols, folds


def run_cross_validation(df: pd.DataFrame, feature_cols: list, folds: list):
    """
    Train + evaluate Random Forest on each fold.

    Returns:
        fold_results: list of dicts with per-fold metrics
        oof_true, oof_pred: pooled out-of-fold true/predicted labels
                             (used for an honest overall confusion matrix)
    """
    fold_results = []
    oof_true = []
    oof_pred = []

    for i, (train_idx, val_idx) in enumerate(folds):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]["label"]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx]["label"]

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")

        fold_results.append({
            "fold": i,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })

        print(f"Fold {i}: accuracy={acc:.4f}, f1_macro={f1_macro:.4f} "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        oof_true.extend(y_val.tolist())
        oof_pred.extend(y_pred.tolist())

    return fold_results, np.array(oof_true), np.array(oof_pred)


def summarize_results(fold_results: list):
    """Print averaged metrics across folds with std deviation (stability check)."""
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY (Random Forest)")
    print("=" * 60)
    print(f"Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"F1 (macro): {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    if np.std(accs) > 0.05:
        print("\nNOTE: High variance across folds (std > 0.05). This can mean "
              "some sessions are harder to generalize to than others — worth "
              "investigating which fold performed worst.")


def plot_confusion_matrix(oof_true, oof_pred, label_names=None, save_path=None):
    """
    Plot a confusion matrix from pooled out-of-fold predictions.
    This is a more honest picture than any single fold, since every row
    was predicted while NOT in that model's training set.
    """
    labels = sorted(np.unique(np.concatenate([oof_true, oof_pred])))
    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest — Out-of-Fold Confusion Matrix (row-normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix plot to {save_path}")
    plt.show()


def print_classification_report(oof_true, oof_pred):
    print("\n" + "=" * 60)
    print("PER-CLASS CLASSIFICATION REPORT (pooled out-of-fold)")
    print("=" * 60)
    print(classification_report(oof_true, oof_pred, digits=3))


def train_final_model(df: pd.DataFrame, feature_cols: list):
    """
    Train the final Random Forest on ALL train_val data (all folds combined).
    This is the model version we'll compare against XGBoost and eventually
    evaluate once on held_out_test.csv.
    """
    X = df[feature_cols]
    y = df["label"]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_DIR / "random_forest_baseline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    print(f"\nFinal Random Forest trained on all {len(df)} train_val rows.")
    print(f"Saved model to {model_path}")

    # Feature importance — useful for your report / interview talking points
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 most important features:")
    print(top_features)

    return model, top_features


def save_cv_summary(fold_results: list):
    """Save per-fold metrics to CSV for later comparison against XGBoost."""
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(fold_results)
    out_path = RESULTS_OUTPUT_DIR / "rf_cv_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved fold-by-fold results to {out_path}")


if __name__ == "__main__":
    df, feature_cols, folds = load_data()

    fold_results, oof_true, oof_pred = run_cross_validation(df, feature_cols, folds)
    summarize_results(fold_results)
    save_cv_summary(fold_results)

    print_classification_report(oof_true, oof_pred)
    plot_confusion_matrix(
        oof_true, oof_pred,
        save_path=str(RESULTS_OUTPUT_DIR / "rf_confusion_matrix.png")
    )

    final_model, top_features = train_final_model(df, feature_cols)