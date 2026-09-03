"""
scripts/explore_rep_counting.py

Day 12 - Step A: Sanity-check rep counting on our own phone recording (10 squats).
This is an exploration script only - not part of the production pipeline yet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# ---- Paths ----
DATA_DIR = Path("data/sample_upload")
ACC_PATH = DATA_DIR / "my_test_session_v2_acc.csv"
GYRO_PATH = DATA_DIR / "my_test_session_v2_gyro.csv"

TRUE_REPS = 10  # ground truth: you counted 10 squats

# ---- Load (no header in these files - see sensorlogger_to_upload_csv.py) ----
col_names = ["frame", "timestamp_ms", "x", "y", "z"]

acc = pd.read_csv(ACC_PATH, header=None, names=col_names)
gyro = pd.read_csv(GYRO_PATH, header=None, names=col_names)

print(f"Accelerometer samples: {len(acc)}")
print(f"Gyroscope samples: {len(gyro)}")
print(f"Acc duration: {(acc['timestamp_ms'].iloc[-1] - acc['timestamp_ms'].iloc[0]) / 1000:.1f} s")

# ---- Compute magnitude signals ----
acc_mag = np.sqrt(acc["x"]**2 + acc["y"]**2 + acc["z"]**2)
gyro_mag = np.sqrt(gyro["x"]**2 + gyro["y"]**2 + gyro["z"]**2)

# ---- Estimate sampling rate (needed to set find_peaks 'distance' sensibly) ----
acc_fs = 1000 / acc["timestamp_ms"].diff().median()  # samples per second
gyro_fs = 1000 / gyro["timestamp_ms"].diff().median()
print(f"Estimated acc sample rate: {acc_fs:.1f} Hz")
print(f"Estimated gyro sample rate: {gyro_fs:.1f} Hz")

# ---- Peak detection ----
# distance: minimum samples between peaks. A squat rep is rarely faster than
# ~1.5s, so we set a minimum spacing of ~1s worth of samples to avoid
# double-counting noise within a single rep.
# prominence: how much a peak must stick out from surrounding signal -
# filters out small jitter that isn't a real rep.

# ---- Parameter sweep: distance (in seconds) between allowed peaks ----
# Rationale: squats produce 2 magnitude spikes per rep (down + up transition).
# We need `distance` large enough to merge those two into one detected peak,
# but small enough not to merge two different reps together.

print("\n--- Distance sweep (seconds) ---")
for dist_sec in [1.0, 1.5, 2.0, 2.5, 3.0]:
    acc_dist = int(acc_fs * dist_sec)
    gyro_dist = int(gyro_fs * dist_sec)

    acc_pk, _ = find_peaks(acc_mag, distance=acc_dist, prominence=acc_mag.std() * 0.5)
    gyro_pk, _ = find_peaks(gyro_mag, distance=gyro_dist, prominence=gyro_mag.std() * 0.5)

    print(f"distance={dist_sec}s -> acc peaks: {len(acc_pk):2d} | gyro peaks: {len(gyro_pk):2d}")

print(f"\nTrue reps (ground truth): {TRUE_REPS}")

# ---- Lock in the winning distance from the sweep, compute final peaks ----
CHOSEN_DISTANCE_SEC = 2.0  # gyro hit exact match (10/10) at this value

acc_min_distance = int(acc_fs * CHOSEN_DISTANCE_SEC)
gyro_min_distance = int(gyro_fs * CHOSEN_DISTANCE_SEC)

acc_peaks, acc_props = find_peaks(
    acc_mag,
    distance=acc_min_distance,
    prominence=acc_mag.std() * 0.5
)
gyro_peaks, gyro_props = find_peaks(
    gyro_mag,
    distance=gyro_min_distance,
    prominence=gyro_mag.std() * 0.5
)

print(f"\n--- Final (distance={CHOSEN_DISTANCE_SEC}s) ---")
print(f"Acc-magnitude peak count:  {len(acc_peaks)}")
print(f"Gyro-magnitude peak count: {len(gyro_peaks)}")