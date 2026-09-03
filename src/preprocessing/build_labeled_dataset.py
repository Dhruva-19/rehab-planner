"""
Day 3 - Step 2: Downsampling + Saving the Labeled Dataset
-------------------------------------------------------------
Goal: fix the non_activity imbalance found in Step 1 (75% of all windows)
by capping it at a multiple of the largest EXERCISE class, then save the
final labeled window dataset to disk so we don't have to re-run the full
raw -> resample -> window -> label pipeline every time we touch feature
engineering or model training.

Strategy (locked in on Day 3):
  - Keep ALL exercise-class windows (they're the valuable, scarce data).
  - Randomly downsample non_activity down to NON_ACTIVITY_CAP_MULTIPLIER x
    the size of the largest exercise class, preserving diversity in rest
    data without letting it dominate training.
  - Residual imbalance across the 10 exercise classes (e.g. jumping_jacks
    337 vs lunges 1103) is deliberately NOT fixed here -> handled later via
    class_weight='balanced' at model training time.

Output: data/processed/labeled_windows.npz containing:
    X           - (N, 600, 6) float32, the raw sensor windows
    y           - (N,) int32, encoded class index (see label_names)
    sessions    - (N,) <U3, which session (w00..w20) each window came from
                  (kept so a future train/test split can be done BY SESSION,
                  not by random window, to avoid data leakage between
                  overlapping windows of the same set of reps)
    purity      - (N,) float32, purity score computed during label alignment
    label_names - (11,) <U30, class index -> name mapping

Run from your project root (same folder as label_alignment.py).
"""

import numpy as np
from pathlib import Path
from collections import Counter

from label_alignment import process_all_sessions, align_session, ACTIONS, PURITY_THRESHOLD

OUTPUT_PATH = Path("data/processed/labeled_windows.npz")
NON_ACTIVITY_CAP_MULTIPLIER = 2.5  # cap non_activity at 2.5x the largest exercise class
RANDOM_SEED = 42                    # fixed seed -> reproducible downsampling


def collect_labeled_windows() -> list:
    """Re-run label alignment (Day 3 Step 1) and return one flat list of labeled windows."""
    all_windows = process_all_sessions()
    labeled = []
    for session, session_windows in all_windows.items():
        for w in align_session(session, session_windows):
            if w["purity"] >= PURITY_THRESHOLD:
                w["session"] = session
                labeled.append(w)
    return labeled


def downsample_non_activity(labeled: list, cap_multiplier: float, seed: int) -> list:
    """
    Cap non_activity windows at cap_multiplier x the largest exercise class,
    keeping ALL exercise-class windows untouched. Downsampling is random
    but reproducible (fixed seed).
    """
    rng = np.random.default_rng(seed)

    exercise = [w for w in labeled if w["label"] != "non_activity"]
    rest = [w for w in labeled if w["label"] == "non_activity"]

    class_counts = Counter(w["label"] for w in exercise)
    largest_exercise_count = max(class_counts.values())
    cap = int(largest_exercise_count * cap_multiplier)

    original_rest_count = len(rest)
    if len(rest) > cap:
        keep_idx = rng.choice(len(rest), size=cap, replace=False)
        rest = [rest[i] for i in keep_idx]

    print(f"Largest exercise class: {largest_exercise_count} windows")
    print(f"non_activity: {original_rest_count} -> {len(rest)} "
          f"(cap = {cap_multiplier}x largest exercise class = {cap})")

    combined = exercise + rest
    rng.shuffle(combined)
    return combined


def save_dataset(labeled: list, output_path: Path):
    """Pack the labeled window list into arrays and save as a single .npz file."""
    label_to_idx = {name: i for i, name in enumerate(ACTIONS)}

    X = np.stack([w["data"] for w in labeled]).astype(np.float32)             # (N, 600, 6)
    y = np.array([label_to_idx[w["label"]] for w in labeled], dtype=np.int32)  # (N,)
    sessions = np.array([w["session"] for w in labeled])                       # (N,)
    purity = np.array([w["purity"] for w in labeled], dtype=np.float32)        # (N,)
    label_names = np.array(ACTIONS)                                            # (11,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X, y=y, sessions=sessions, purity=purity, label_names=label_names,
    )
    print(f"\nSaved {len(labeled)} windows -> {output_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


def main():
    labeled = collect_labeled_windows()
    print(f"Total labeled windows (post purity filter): {len(labeled)}\n")

    balanced = downsample_non_activity(labeled, NON_ACTIVITY_CAP_MULTIPLIER, RANDOM_SEED)

    print("\nFinal class distribution:")
    for cls, count in Counter(w["label"] for w in balanced).most_common():
        print(f"  {cls:<28} {count:>6}")

    save_dataset(balanced, OUTPUT_PATH)


if __name__ == "__main__":
    main()
