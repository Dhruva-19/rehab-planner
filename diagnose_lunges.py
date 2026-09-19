"""
Quick diagnostic: print every raw per-window prediction for the original
Lunges capture, to check whether the model is under-detecting active
windows for slow-paced lunges (same root cause already diagnosed for
squats), vs. something else (e.g. a genuinely short active burst).

Run from project root.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path("src")
sys.path.append(str(_PROJECT_ROOT / "preprocessing"))
sys.path.append(str(_PROJECT_ROOT / "feature_engineering"))
sys.path.append(str(_PROJECT_ROOT / "inference"))
sys.path.append(str(_PROJECT_ROOT / "feedback"))

from predict_pipeline import predict_from_raw_csv
from aggregate_sets import aggregate_into_sets

ACC = "data/sample_upload/Lunges_acc.csv"
GYRO = "data/sample_upload/Lunges_gyro.csv"

results = predict_from_raw_csv(ACC, GYRO)
print(f"\n{len(results)} raw windows:\n")
print(results.to_string(index=False))

sets = aggregate_into_sets(results)
print(f"\n{len(sets)} sets after aggregation:\n")
print(sets.to_string(index=False))
