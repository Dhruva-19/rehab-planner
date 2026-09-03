"""One-off: extract just label_names from the full labeled_windows.npz
into a small standalone file, so it can be committed to the deploy repo
without shipping the full (large) windowed dataset."""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
full_npz = PROJECT_ROOT / "data" / "processed" / "labeled_windows.npz"
out_path = PROJECT_ROOT / "saved_models" / "label_names.npz"

data = np.load(full_npz, allow_pickle=True)
np.savez(out_path, label_names=data["label_names"])

print(f"Saved label_names to {out_path}")
print(f"Contents: {np.load(out_path, allow_pickle=True)['label_names']}")