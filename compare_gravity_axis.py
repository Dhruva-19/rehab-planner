"""
compare_gravity_axis.py

Compares the gravity axis convention in MM-Fit's TRAINING data
(right-thigh-pocket, sp_r) against a live phone capture, to check
for a coordinate-system mismatch between the browser DeviceMotion
capture and MM-Fit's original Android sensor axes.

Run from project root:
    python compare_gravity_axis.py
"""
import numpy as np

WINDOWS_PATH = "data/processed/labeled_windows.npz"  # adjust if path differs
NON_ACTIVITY_LABEL = 10

data = np.load(WINDOWS_PATH, allow_pickle=True)
X = data["X"]          # (N, 600, 6) -> [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
y = data["y"]

rest_mask = (y == NON_ACTIVITY_LABEL)
rest_windows = X[rest_mask]  # (M, 600, 6)

acc_rest = rest_windows[:, :, 0:3].reshape(-1, 3)  # flatten all rest samples

axis_means = acc_rest.mean(axis=0)
axis_stds = acc_rest.std(axis=0)
magnitude = np.linalg.norm(acc_rest, axis=1)

print(f"Training non_activity windows: {rest_mask.sum()} windows, "
      f"{acc_rest.shape[0]} total samples\n")
print(f"Per-axis mean (ax, ay, az): {axis_means.round(3)}")
print(f"Per-axis std  (ax, ay, az): {axis_stds.round(3)}")
print(f"Magnitude — mean: {magnitude.mean():.3f}, std: {magnitude.std():.3f}")

dominant_axis = np.argmax(np.abs(axis_means))
axis_labels = ["X", "Y", "Z"]
print(f"\nDominant gravity axis in TRAINING data: {axis_labels[dominant_axis]} "
      f"(mean = {axis_means[dominant_axis]:.3f})")
print("\nCompare this to the live capture's in-pocket steady state: "
      "Y-axis, mean ~ -8.8 to -9.3")    