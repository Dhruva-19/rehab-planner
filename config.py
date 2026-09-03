"""
config.py
---------
Central place for all file paths and project-wide constants.
Import this everywhere instead of hardcoding paths, so the project
still works if you move folders around later.
"""

from pathlib import Path

# ----- Base directories -----
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# MM-Fit specific: after download, extract the dataset here so that
# RAW_DATA_DIR / "mm-fit" / "w00" / "w00_accelerometer.npy" etc. resolves.
MMFIT_DIR = RAW_DATA_DIR / "mm-fit"

SRC_DIR = ROOT_DIR / "src"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
DATABASE_DIR = ROOT_DIR / "database"

# ----- Database -----
DB_PATH = DATABASE_DIR / "rehab_sessions.db"

# ----- Sensor / modeling constants -----
# MM-Fit ships multiple modalities; we start with wrist accelerometer +
# gyroscope since that best simulates a single rehab wearable.
TARGET_SAMPLING_RATE_HZ = 50   # resample all signals to this rate
WINDOW_LENGTH_SEC = 3          # sliding window length for feature extraction
WINDOW_STRIDE_SEC = 0.5        # stride between consecutive windows

# MM-Fit activity labels (10 exercise classes + background/null class,
# confirmed against the official mm-fit GitHub repo's utils/dataset.py)
MMFIT_ACTIVITIES = [
    "squats", "lunges", "bicep_curls", "situps", "pushups",
    "tricep_extensions", "dumbbell_rows", "jumping_jacks",
    "dumbbell_shoulder_press", "lateral_shoulder_raises", "non_activity"
]

# Official MM-Fit train/val/test split (session IDs), recommended by the
# dataset authors for comparable benchmarking. 'unseen_test' holds out
# entire subjects for cross-subject evaluation.
MMFIT_SPLIT = {
    "train": ["w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"],
    "val": ["w14", "w15", "w19"],
    "test": ["w09", "w10", "w11"],
    "unseen_test": ["w00", "w05", "w12", "w13", "w20"],
}

# Ensure key directories exist when this module is imported
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, DATABASE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
