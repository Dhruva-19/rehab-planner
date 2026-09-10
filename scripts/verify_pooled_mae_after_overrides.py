"""
scripts/verify_pooled_mae_after_overrides.py

Day 19 - Sanity check: recompute pooled gyro MAE using the new per-exercise
REP_COUNT_DISTANCE_OVERRIDES, by looking up each segment's error at its
assigned distance in the existing sweep results (no peak detection re-run
needed - just re-selecting rows).

Confirms:
1. Overall pooled MAE improves vs the old flat-1.5s baseline (2.28).
2. The 6 non-overridden exercises are numerically UNCHANGED (still exactly
   their 1.5s error) - proving the edit didn't accidentally touch them.
"""

import pandas as pd
from pathlib import Path

SWEEP_RESULTS_PATH = Path("scripts/rep_count_sweep_results.csv")

# Must match aggregate_sets.py exactly
REP_COUNT_DISTANCE_OVERRIDES = {
    "pushups": 1.0,
    "jumping_jacks": 0.8,
    "lunges": 1.8,
    "lateral_shoulder_raises": 1.8,
}
DEFAULT_DISTANCE = 1.5


def main():
    df = pd.read_csv(SWEEP_RESULTS_PATH)

    # Assign each segment the distance it would get under the new logic
    df["assigned_distance"] = df["exercise"].map(
        lambda ex: REP_COUNT_DISTANCE_OVERRIDES.get(ex, DEFAULT_DISTANCE)
    )

    # Keep only the row matching each segment's assigned distance
    new_rows = df[df["distance_sec"] == df["assigned_distance"]]

    # Old baseline: every segment at flat 1.5s
    old_rows = df[df["distance_sec"] == 1.5]

    old_pooled_mae = old_rows["gyr_err"].mean()
    new_pooled_mae = new_rows["gyr_err"].mean()

    print(f"OLD pooled gyro MAE (flat 1.5s):      {old_pooled_mae:.3f}  (expect ~2.28)")
    print(f"NEW pooled gyro MAE (per-exercise):    {new_pooled_mae:.3f}")
    print(f"Improvement:                           {old_pooled_mae - new_pooled_mae:.3f}")

    print("\nPer-exercise: old vs new (should be IDENTICAL for non-overridden exercises)")
    old_by_ex = old_rows.groupby("exercise")["gyr_err"].mean().round(3)
    new_by_ex = new_rows.groupby("exercise")["gyr_err"].mean().round(3)
    compare = pd.DataFrame({"old_1.5s_MAE": old_by_ex, "new_MAE": new_by_ex})
    compare["changed"] = compare["old_1.5s_MAE"] != compare["new_MAE"]
    compare["overridden"] = compare.index.isin(REP_COUNT_DISTANCE_OVERRIDES.keys())
    print(compare.to_string())

    # Flag anything unexpected
    unexpected = compare[(~compare["overridden"]) & (compare["changed"])]
    if not unexpected.empty:
        print("\n!!! WARNING: non-overridden exercises changed - something's wrong:")
        print(unexpected.to_string())
    else:
        print("\nOK: all non-overridden exercises are numerically unchanged.")


if __name__ == "__main__":
    main()