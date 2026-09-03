"""
src/model/train_xgboost.py

Purpose: Train an XGBoost baseline using the SAME session-aware CV folds
as the Random Forest baseline, so results are directly comparable.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
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

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": 42,
}


def load_data():
    """Load the train_val feature set and the pre-computed CV fold indices."""
    df = pd.read_csv(TRAIN_VAL_PATH)

    with open(FOLDS_PATH, "r") as f:
        fold_dict = json.load(f)

    folds = [
        (np.array(fold_dict[f"fold_{i}"]["train_idx"]),
         np.array(fold_dict[f"fold_{i}"]["val_idx"]))
        for i in range(len(fold_dict))
    ]

    feature_cols = [c for c in df.columns if c not in ("label", "session")]
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, {len(folds)} folds.")
    return df, feature_cols, folds


def run_cross_validation(df: pd.DataFrame, feature_cols: list, folds: list, label_encoder: LabelEncoder):
    """
    Train + evaluate XGBoost on each fold.

    Note: XGBoost's multi:softprob objective requires integer labels
    (0..N-1), so we use the label_encoder to convert your class labels
    (which may be 0-10 already, or strings) into that form.
    """
    fold_results = []
    oof_true = []
    oof_pred = []

    y_all_encoded = label_encoder.transform(df["label"])

    for i, (train_idx, val_idx) in enumerate(folds):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = y_all_encoded[train_idx]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = y_all_encoded[val_idx]

        model = XGBClassifier(**XGB_PARAMS)
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
    print("CROSS-VALIDATION SUMMARY (XGBoost)")
    print("=" * 60)
    print(f"Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"F1 (macro): {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    if np.std(accs) > 0.05:
        print("\nNOTE: High variance across folds (std > 0.05) — same caveat "
              "as Random Forest, some sessions are harder to generalize to.")


def plot_confusion_matrix(oof_true, oof_pred, label_encoder: LabelEncoder, save_path=None):
    """
    Plot a confusion matrix from pooled out-of-fold predictions, using
    the ORIGINAL class labels (not encoded integers) for readability.
    """
    oof_true_labels = label_encoder.inverse_transform(oof_true)
    oof_pred_labels = label_encoder.inverse_transform(oof_pred)

    labels = sorted(np.unique(np.concatenate([oof_true_labels, oof_pred_labels])))
    cm = confusion_matrix(oof_true_labels, oof_pred_labels, labels=labels)
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
    plt.title("XGBoost — Out-of-Fold Confusion Matrix (row-normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix plot to {save_path}")
    plt.show()


def print_classification_report(oof_true, oof_pred, label_encoder: LabelEncoder):
    oof_true_labels = label_encoder.inverse_transform(oof_true)
    oof_pred_labels = label_encoder.inverse_transform(oof_pred)

    print("\n" + "=" * 60)
    print("PER-CLASS CLASSIFICATION REPORT (pooled out-of-fold)")
    print("=" * 60)
    print(classification_report(oof_true_labels, oof_pred_labels, digits=3))


def train_final_model(df: pd.DataFrame, feature_cols: list, label_encoder: LabelEncoder):
    """
    Train the final XGBoost on ALL train_val data (all folds combined).
    """
    X = df[feature_cols]
    y = label_encoder.transform(df["label"])

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_OUTPUT_DIR / "xgboost_v2_orientation_features.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "label_encoder": label_encoder
        }, f)

    print(f"\nFinal XGBoost trained on all {len(df)} train_val rows.")
    print(f"Saved model to {model_path}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 most important features:")
    print(top_features)

    return model, top_features


def save_cv_summary(fold_results: list):
    """Save per-fold metrics to CSV for later comparison against Random Forest."""
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(fold_results)
    out_path = RESULTS_OUTPUT_DIR / "xgb_v2_cv_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved fold-by-fold results to {out_path}")


if __name__ == "__main__":
    df, feature_cols, folds = load_data()

    # XGBoost needs integer-encoded labels (0..N-1), fit encoder on ALL
    # labels up front so train/val folds share a consistent mapping.
    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    print(f"Label classes: {list(label_encoder.classes_)}")

    fold_results, oof_true, oof_pred = run_cross_validation(df, feature_cols, folds, label_encoder)
    summarize_results(fold_results)
    save_cv_summary(fold_results)

    print_classification_report(oof_true, oof_pred, label_encoder)
    plot_confusion_matrix(
        oof_true, oof_pred, label_encoder,
        save_path=str(RESULTS_OUTPUT_DIR / "xgb_v2_confusion_matrix.png")
    )

    final_model, top_features = train_final_model(df, feature_cols, label_encoder)