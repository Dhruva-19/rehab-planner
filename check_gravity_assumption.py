"""
check_gravity_assumption.py

Purpose: Diagnostic for the Day 11 phone-recording misclassification bug.
Checks whether MM-Fit's raw sp_r_acc.npy includes gravity (~9.8 m/s^2
magnitude at rest) or excludes it (~0 m/s^2 at rest, like Sensor
Logger's Accelerometer.csv). This tells us whether the converter script's
choice of "Accelerometer.csv over TotalAcceleration.csv" was correct.

Run this from your project root:
    python check_gravity_assumption.py

It will look for w00's sp_r_acc.npy under data/raw/mm-fit/ — adjust
MMFIT_ROOT below if your folder layout differs.
"""

import numpy as np
from pathlib import Path

MMFIT_ROOT = Path("data/raw/mm-fit")
SESSION = "w00"


def find_sp_r_acc_file(session: str) -> Path:
    """Locate the sp_r_acc .npy file for a given session, trying a couple
    of common MM-Fit folder layouts."""
    candidates = [
        MMFIT_ROOT / session / f"{session}_sp_r_acc.npy",
        MMFIT_ROOT / session / "sp_r_acc.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to a recursive search
    matches = list(MMFIT_ROOT.rglob(f"*{session}*sp_r_acc*.npy"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Couldn't find sp_r_acc.npy for session {session} under {MMFIT_ROOT}. "
        f"Edit MMFIT_ROOT / SESSION at the top of this script to match your layout."
    )


def main():
    acc_path = find_sp_r_acc_file(SESSION)
    print(f"Loading: {acc_path}")

    data = np.load(acc_path)
    print(f"Shape: {data.shape}")

    # MM-Fit raw arrays are typically [frame, timestamp, x, y, z] or [timestamp, x, y, z]
    # We assume the LAST 3 columns are x, y, z regardless of layout.
    xyz = data[:, -3:]

    # Per-axis mean (tells us if gravity sits on one consistent axis)
    axis_means = xyz.mean(axis=0)
    print(f"\nPer-axis mean (x, y, z): {axis_means}")

    # Magnitude tells us if gravity is present regardless of axis orientation
    magnitude = np.linalg.norm(xyz, axis=1)
    print(f"Magnitude — mean: {magnitude.mean():.3f}, "
          f"median: {np.median(magnitude):.3f}, "
          f"std: {magnitude.std():.3f}")
    print(f"Magnitude — min: {magnitude.min():.3f}, max: {magnitude.max():.3f}")

    print("\n--- Interpretation ---")
    if magnitude.mean() > 5.0:
        print("Mean magnitude is close to ~9.8 -> MM-Fit's sp_r_acc INCLUDES gravity.")
        print("This means our phone conversion is WRONG: we should have used")
        print("TotalAcceleration.csv (gravity included) from Sensor Logger, not")
        print("Accelerometer.csv (gravity excluded).")
    elif magnitude.mean() < 2.0:
        print("Mean magnitude is close to ~0 -> MM-Fit's sp_r_acc EXCLUDES gravity.")
        print("This matches our converter's assumption (Accelerometer.csv was correct).")
        print("The non_activity misclassification likely has a different cause —")
        print("check sampling rate, axis order/orientation, or units (g vs m/s^2) next.")
    else:
        print("Ambiguous magnitude — inspect the raw values below manually.")

    print("\nFirst 5 rows (raw):")
    print(data[:5])


if __name__ == "__main__":
    main()
