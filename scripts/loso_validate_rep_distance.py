"""
scripts/loso_validate_rep_distance.py

Day 19 - Leave-one-session-out (LOSO) validation of per-exercise rep-counting
distances.

Reuses the long-format sweep results already computed in
scripts/rep_count_sweep_results.csv (session, exercise, distance_sec, gyr_err
per segment) instead of recomputing peak detection - this only regroups
existing numbers, so it runs instantly.

For each exercise, for each held-out session S:
    - "train" distance is chosen by minimizing mean gyr_err over the OTHER
      4 sessions' segments for that exercise (this is what "best_distance"
      per exercise SHOULD look like if we didn't get to see session S).
    - that distance's error is then measured only on session S's segments
      for that exercise (never seen during distance selection).
Rotate S across all 5 sessions and pool the out-of-fold errors -> LOSO MAE.

This tells us whether the per-exercise "best distance" numbers from the
in-sample sweep generalize, or are overfit to noise in a small (~15-16
segment) per-exercise sample.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SWEEP_RESULTS_PATH = Path("scripts/rep_count_sweep_results.csv")


def main():
    df = pd.read_csv(SWEEP_RESULTS_PATH)

    results = []
    for exercise, ex_df in df.groupby("exercise"):
        sessions_present = ex_df["session"].unique().tolist()

        oof_errors = []        # LOSO out-of-fold errors, pooled
        chosen_distances = []  # which distance got picked each fold, for transparency

        for held_out in sessions_present:
            train_df = ex_df[ex_df["session"] != held_out]
            test_df = ex_df[ex_df["session"] == held_out]

            if train_df.empty or test_df.empty:
                continue

            # Pick the distance that minimizes mean gyr_err on the TRAIN sessions only
            train_mae_by_distance = train_df.groupby("distance_sec")["gyr_err"].mean()
            best_train_distance = train_mae_by_distance.idxmin()
            chosen_distances.append(best_train_distance)

            # Evaluate that distance on the held-out session's segments (never used to pick it)
            test_rows = test_df[test_df["distance_sec"] == best_train_distance]
            oof_errors.extend(test_rows["gyr_err"].tolist())

        if not oof_errors:
            continue

        # For comparison: in-sample "best" (Day 19's first sweep - picked
        # AND evaluated on the same full data, so optimistic)
        full_mae_by_distance = ex_df.groupby("distance_sec")["gyr_err"].mean()
        in_sample_best_distance = full_mae_by_distance.idxmin()
        in_sample_best_mae = full_mae_by_distance.min()

        # Baseline: current locked global distance
        mae_at_1_5 = ex_df[ex_df["distance_sec"] == 1.5]["gyr_err"].mean()

        results.append({
            "exercise": exercise,
            "n_segments": ex_df["seg_id"].nunique(),
            "mae_at_1.5s": round(mae_at_1_5, 2),
            "in_sample_best_distance": in_sample_best_distance,
            "in_sample_best_mae": round(in_sample_best_mae, 2),
            "loso_mae": round(float(np.mean(oof_errors)), 2),
            "loso_distances_chosen": sorted(set(chosen_distances)),
        })

    out_df = pd.DataFrame(results).sort_values("loso_mae", ascending=False)
    print(out_df.to_string(index=False))
    out_df.to_csv("scripts/loso_rep_distance_validation.csv", index=False)
    print("\nSaved: scripts/loso_rep_distance_validation.csv")

    print("\n" + "=" * 60)
    print("Interpretation guide:")
    print("- loso_mae close to in_sample_best_mae -> distance genuinely")
    print("  generalizes, safe to lock in.")
    print("- loso_mae much closer to (or worse than) mae_at_1.5s -> the")
    print("  in-sample 'best' was overfitting; don't trust it as-is.")
    print("- loso_distances_chosen with >1 value -> the optimal distance")
    print("  is unstable across folds, another overfitting signal.")


if __name__ == "__main__":
    main()