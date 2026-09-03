"""
src/feedback/quality_scorer.py

Purpose: Take the per-set DataFrame produced by aggregate_sets.aggregate_into_sets
and attach a heuristic quality score (0-100) and a human-readable feedback
message to each exercise set.

Design note (documented for viva): MM-Fit has no ground-truth "quality"
labels -- it's a recognition dataset, not a form-quality dataset. So this
is a deliberately RULE-BASED / heuristic scorer, not a trained model. It
combines three signals that are already available after set aggregation:

    1. confidence_score   - how sure the model was, on average, that this
                             set is what it says it is (from mean_confidence),
                             penalized if confidence swung wildly within the
                             set (from confidence_std).
    2. stability_score     - raw_agreement: fraction of raw, pre-smoothing
                             window predictions that already agreed with the
                             set's final label. Low = the movement pattern
                             was inconsistent enough to cause label flicker
                             before smoothing cleaned it up.
    3. completeness_score  - 1.0 if the set met min_set_windows, 0.5 if it
                             was flagged is_short (possible cut-off rep or
                             genuinely brief set -- can't tell which from
                             sensor data alone, so it's penalized, not
                             disqualified).

This is intentionally explainable rather than "black box": every score can
be traced back to a specific, inspectable number already sitting in the
sets DataFrame.
"""

import numpy as np
import pandas as pd

# Weights are a starting point, tuned for interpretability, not fit to data
# (there's no quality ground truth to fit them to). Documented here so they
# can be justified/tuned later.
WEIGHT_CONFIDENCE = 0.5
WEIGHT_STABILITY = 0.3
WEIGHT_COMPLETENESS = 0.2

SHORT_SET_COMPLETENESS_PENALTY = 0.5  # completeness_score for is_short sets

# Confidence-std penalty: how much to subtract from confidence_score per
# unit of confidence_std, capped so it can never push confidence_score
# below 0.
CONFIDENCE_STD_PENALTY_FACTOR = 1.0

# Score bands -> feedback message. Checked top-down, first match wins.
FEEDBACK_BANDS = [
    (85, "Good form and consistency."),
    (60, "Some inconsistency detected in this set — worth a closer look."),
    (0,  "Movement pattern was unstable — review form or redo this set."),
]


def _confidence_score(mean_confidence: float, confidence_std: float) -> float:
    """
    Mean confidence, penalized for instability within the set.
    confidence_std is NaN for single-window sets -> treated as 0
    (nothing to compare within a single window, so no penalty).
    """
    std = 0.0 if pd.isna(confidence_std) else confidence_std
    score = mean_confidence - CONFIDENCE_STD_PENALTY_FACTOR * std
    return float(np.clip(score, 0.0, 1.0))


def _completeness_score(is_short: bool) -> float:
    return SHORT_SET_COMPLETENESS_PENALTY if is_short else 1.0


def _feedback_message(score: float) -> str:
    for threshold, message in FEEDBACK_BANDS:
        if score >= threshold:
            return message
    return FEEDBACK_BANDS[-1][1]  # unreachable given threshold 0, but safe


def score_sets(sets_df: pd.DataFrame,
               exclude_labels=("non_activity",)) -> pd.DataFrame:
    """
    Attach quality_score and feedback columns to a sets DataFrame.

    Parameters
    ----------
    sets_df : DataFrame, output of aggregate_sets.aggregate_into_sets.
        Must contain: label, mean_confidence, confidence_std,
        raw_agreement, is_short.
    exclude_labels : labels to skip scoring for (default: non_activity,
        since "exercise quality" doesn't apply to rest periods). Excluded
        rows get quality_score = NaN and feedback = "Not scored (rest period)".

    Returns
    -------
    DataFrame: sets_df with two new columns appended:
        quality_score : float, 0-100, NaN for excluded labels
        feedback      : str, human-readable message
    """
    df = sets_df.copy()

    confidence_scores = df.apply(
        lambda r: _confidence_score(r["mean_confidence"], r["confidence_std"]),
        axis=1
    )
    stability_scores = df["raw_agreement"].astype(float)
    completeness_scores = df["is_short"].apply(_completeness_score)

    raw_scores = 100 * (
        WEIGHT_CONFIDENCE * confidence_scores +
        WEIGHT_STABILITY * stability_scores +
        WEIGHT_COMPLETENESS * completeness_scores
    )

    is_excluded = df["label"].isin(exclude_labels)

    df["quality_score"] = np.where(is_excluded, np.nan, raw_scores)
    df["feedback"] = np.where(
        is_excluded,
        "Not scored (rest period).",
        [ _feedback_message(s) for s in raw_scores ]
    )

    # Round score for display; keep NaN as NaN
    df["quality_score"] = df["quality_score"].round(1)

    return df


def session_summary(scored_df: pd.DataFrame) -> dict:
    """
    Simple session-level summary: average quality score across scored
    (non-excluded) sets, plus counts, for a dashboard header/overview.
    """
    scored = scored_df.dropna(subset=["quality_score"])
    if scored.empty:
        return {"avg_quality_score": None, "num_scored_sets": 0,
                "num_short_sets": int(scored_df["is_short"].sum())}
    return {
        "avg_quality_score": round(float(scored["quality_score"].mean()), 1),
        "num_scored_sets": int(len(scored)),
        "num_short_sets": int(scored_df["is_short"].sum()),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parents[1] / "inference"))
    from aggregate_sets import aggregate_into_sets
    from predict_pipeline import predict_from_raw_csv

    ACC_CSV = "data/sample_upload/session_acc.csv"
    GYRO_CSV = "data/sample_upload/session_gyro.csv"

    window_results = predict_from_raw_csv(ACC_CSV, GYRO_CSV)
    session_sets = aggregate_into_sets(window_results)
    scored = score_sets(session_sets)

    print(scored.to_string(index=False))
    print()
    print("Session summary:", session_summary(scored))
    