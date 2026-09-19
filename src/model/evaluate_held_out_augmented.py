"""
src/model/evaluate_held_out_augmented.py

Day 25 - Step: evaluate the augmented model (xgboost_v4_augmented.pkl)
against the SAME held-out MM-Fit test sessions used for the original
v2 baseline (accuracy 0.8338, F1 macro 0.8359), so the comparison is
apples-to-apples.

Reuses evaluate() and plot_confusion_matrix() from evaluate_held_out.py
directly, unchanged. Deliberately does NOT overwrite the original
final_held_out_*.csv/.txt/.png files - those are the official v2
baseline artifacts for the report ("Run this ONCE" per that file's own
docstring). This writes to separate *_v4_augmented files instead, so
both results stay available side by side.

Run from your project root, same as evaluate_held_out.py.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from evaluate_held_out import evaluate, plot_confusion_matrix

HELD_OUT_TEST_PATH = "data/processed/held_out_test.csv"
NEW_MODEL_PATH = "saved_models/xgboost_v4_augmented.pkl"
RESULTS_OUTPUT_DIR = Path("data/processed")

# For the printed comparison line - update if your true baseline differs.
BASELINE_ACCURACY = 0.8338
BASELINE_F1_MACRO = 0.8359


def load_model_and_data():
    with open(NEW_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    feature_cols = saved["feature_cols"]
    label_encoder = saved["label_encoder"]

    df_test = pd.read_csv(HELD_OUT_TEST_PATH)

    missing_cols = set(feature_cols) - set(df_test.columns)
    if missing_cols:
        raise ValueError(
            f"held_out_test.csv is missing {len(missing_cols)} feature columns "
            f"the model expects, e.g. {list(missing_cols)[:5]}."
        )

    print(f"Loaded model trained on {len(feature_cols)} features.")
    print(f"Held-out test set: {len(df_test)} rows from sessions "
          f"{sorted(df_test['session'].unique())}")

    return model, feature_cols, label_encoder, df_test


def save_summary_augmented(acc, f1_macro, y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    out_path = RESULTS_OUTPUT_DIR / "final_held_out_results_v4_augmented.csv"
    df_report.to_csv(out_path)
    print(f"Saved per-class results to {out_path}")

    summary_path = RESULTS_OUTPUT_DIR / "final_held_out_summary_v4_augmented.txt"
    with open(summary_path, "w") as f:
        f.write("HELD-OUT TEST RESULTS - v4 AUGMENTED (MM-Fit + live captures)\n")
        f.write("Model: XGBoost, v4 augmented (113 features, +114 live windows)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
        f.write(f"\nBaseline (v2, MM-Fit only) for comparison:\n")
        f.write(f"Accuracy: {BASELINE_ACCURACY:.4f}\n")
        f.write(f"F1 (macro): {BASELINE_F1_MACRO:.4f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    model, feature_cols, label_encoder, df_test = load_model_and_data()
    y_test, y_pred, acc, f1_macro = evaluate(model, feature_cols, label_encoder, df_test)

    print("\n" + "=" * 60)
    print("COMPARISON vs v2 baseline (MM-Fit only)")
    print("=" * 60)
    print(f"Baseline  : accuracy={BASELINE_ACCURACY:.4f}, f1_macro={BASELINE_F1_MACRO:.4f}")
    print(f"Augmented : accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")
    print(f"Delta     : accuracy={acc - BASELINE_ACCURACY:+.4f}, "
          f"f1_macro={f1_macro - BASELINE_F1_MACRO:+.4f}")

    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(RESULTS_OUTPUT_DIR / "final_held_out_confusion_matrix_v4_augmented.png")
    )
    save_summary_augmented(acc, f1_macro, y_test, y_pred)
