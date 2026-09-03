"""
src/model/evaluate_held_out.py

Purpose: The ONE honest evaluation of the final locked model
(XGBoost + v2/113-feature set) on the held-out test sessions
(w00, w01, w08, w15, w17) that were never touched during CV or
feature-comparison experiments. Run this ONCE — this number goes
in the final report.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Config
# -----------------------------
HELD_OUT_TEST_PATH = "data/processed/held_out_test.csv"
MODEL_PATH = "saved_models/xgboost_v2_orientation_features.pkl"
RESULTS_OUTPUT_DIR = Path("data/processed")


def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    feature_cols = saved["feature_cols"]
    label_encoder = saved["label_encoder"]

    df_test = pd.read_csv(HELD_OUT_TEST_PATH)

    # Sanity check: confirm the held-out CSV has exactly the columns
    # the model was trained on -- catches a stale/mismatched split file
    # immediately instead of failing with a cryptic sklearn error.
    missing_cols = set(feature_cols) - set(df_test.columns)
    if missing_cols:
        raise ValueError(
            f"held_out_test.csv is missing {len(missing_cols)} feature columns "
            f"the model expects, e.g. {list(missing_cols)[:5]}. "
            f"Did you rerun split_data.py with FEATURES_PATH pointed at "
            f"features_v2.csv before running this script?"
        )

    print(f"Loaded model trained on {len(feature_cols)} features.")
    print(f"Held-out test set: {len(df_test)} rows from sessions "
          f"{sorted(df_test['session'].unique())}")

    return model, feature_cols, label_encoder, df_test


def evaluate(model, feature_cols, label_encoder, df_test):
    X_test = df_test[feature_cols]
    y_test_encoded = label_encoder.transform(df_test["label"])

    y_pred_encoded = model.predict(X_test)

    y_test = label_encoder.inverse_transform(y_test_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("FINAL HELD-OUT TEST RESULTS (unseen sessions)")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS CLASSIFICATION REPORT (held-out test)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))

    return y_test, y_pred, acc, f1_macro


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",   # different color from CV plots, to visually
                          # distinguish "final exam" from "practice" results
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("FINAL MODEL — Held-Out Test Confusion Matrix (unseen sessions)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved confusion matrix plot to {save_path}")
    plt.show()


def save_summary(acc, f1_macro, y_test, y_pred):
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    out_path = RESULTS_OUTPUT_DIR / "final_held_out_results.csv"
    df_report.to_csv(out_path)
    print(f"Saved per-class results to {out_path}")

    summary_path = RESULTS_OUTPUT_DIR / "final_held_out_summary.txt"
    with open(summary_path, "w") as f:
        f.write("FINAL HELD-OUT TEST RESULTS\n")
        f.write(f"Model: XGBoost, v2 feature set (113 features)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    model, feature_cols, label_encoder, df_test = load_model_and_data()
    y_test, y_pred, acc, f1_macro = evaluate(model, feature_cols, label_encoder, df_test)
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(RESULTS_OUTPUT_DIR / "final_held_out_confusion_matrix.png")
    )
    save_summary(acc, f1_macro, y_test, y_pred)