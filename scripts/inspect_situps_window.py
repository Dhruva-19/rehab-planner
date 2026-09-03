"""
Day 13 diagnostic: inspect raw signal magnitude in the 00:54-02:28 window
to check whether situps got fragmented into non_activity, or genuinely
didn't register at all.

Run from project root:
    python scripts/inspect_situps_window.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ACC_PATH = "data/sample_upload/day13_5exercises_acc.csv"
GYRO_PATH = "data/sample_upload/day13_5exercises_gyro.csv"

COLS = ["session_idx", "timestamp_ms", "x", "y", "z"]

acc = pd.read_csv(ACC_PATH, header=None, names=COLS)
gyro = pd.read_csv(GYRO_PATH, header=None, names=COLS)

# Elapsed seconds from the start of the recording (matches app.py's mm:ss)
t0 = min(acc["timestamp_ms"].min(), gyro["timestamp_ms"].min())
acc["t_sec"] = (acc["timestamp_ms"] - t0) / 1000.0
gyro["t_sec"] = (gyro["timestamp_ms"] - t0) / 1000.0

# Magnitudes
acc["mag"] = np.sqrt(acc["x"]**2 + acc["y"]**2 + acc["z"]**2)
gyro["mag"] = np.sqrt(gyro["x"]**2 + gyro["y"]**2 + gyro["z"]**2)

# Window of interest: lunges tail + the long non_activity gap + the
# second "squats" block, per the app's Sets table
WINDOW_START = 45   # a bit before lunges ends, for context
WINDOW_END = 235    # a bit after the second squats-labeled block ends

acc_w = acc[(acc["t_sec"] >= WINDOW_START) & (acc["t_sec"] <= WINDOW_END)]
gyro_w = gyro[(gyro["t_sec"] >= WINDOW_START) & (gyro["t_sec"] <= WINDOW_END)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(acc_w["t_sec"], acc_w["mag"], linewidth=0.8, color="tab:blue")
axes[0].set_ylabel("Acc magnitude (incl. gravity)")
axes[0].set_title("Raw signal magnitude: 00:45-03:55 window")
axes[0].axhline(9.8, color="gray", linestyle=":", linewidth=0.8, label="~gravity baseline")
axes[0].legend(loc="upper right")

axes[1].plot(gyro_w["t_sec"], gyro_w["mag"], linewidth=0.8, color="tab:orange")
axes[1].set_ylabel("Gyro magnitude")
axes[1].set_xlabel("Elapsed time (s)")

# Mark the app's detected segment boundaries for reference
boundaries = {
    "lunges end (~00:55)": 55,
    "non_activity start (00:54)": 54,
    "non_activity end (02:05)": 125,
    "squats#2 start (02:04)": 124,
    "squats#2 end (02:28)": 148,
}
for ax in axes:
    for label, t in boundaries.items():
        ax.axvline(t, color="red", linestyle="--", linewidth=0.7, alpha=0.6)

plt.tight_layout()
out_path = "data/sample_upload/day13_situps_window_check.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
print("\nLook for periodic bursts in the gyro magnitude plot between the")
print("red dashed lines marking 'non_activity start' and 'non_activity end'.")
print("Regular up-down bursts = situps happened but got mislabeled as rest.")
print("Flat/noisy-only signal = situps genuinely didn't produce a detectable pattern.")
