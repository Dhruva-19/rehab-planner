"""
data_loader.py
---------------
Utilities for loading raw MM-Fit sensor data and labels into
pandas DataFrames that the rest of the pipeline (preprocessing,
feature engineering, modeling) can work with directly.

MM-Fit layout (after you extract the downloaded dataset into
data/raw/mm-fit/):

    data/raw/mm-fit/
        w00/
            w00_sw_l_acc.npy   # left smartwatch accelerometer
            w00_sw_l_gyr.npy   # left smartwatch gyroscope
            w00_sw_r_acc.npy   # right smartwatch accelerometer
            w00_sw_r_gyr.npy   # right smartwatch gyroscope
            w00_labels.csv     # (start_frame, end_frame, reps, activity)
            ... other modalities (sp_*, eb_*, pose_2d, pose_3d, etc.)
        w01/
        ...
        w20/

Each sensor modality .npy file has shape (N, 5): columns are
(Frame, Timestamp, X, Y, Z). We use the right smartwatch (sw_r_acc /
sw_r_gyr) by default since it best simulates a single wrist-worn
rehab device, but any modality name can be passed in.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MMFIT_DIR, MMFIT_ACTIVITIES


def list_sessions(mmfit_dir: Path = MMFIT_DIR) -> list:
    """
    Return sorted list of session folder names (e.g. ['w00', 'w01', ...])
    found inside the MM-Fit dataset directory.
    """
    if not mmfit_dir.exists():
        raise FileNotFoundError(
            f"MM-Fit dataset not found at {mmfit_dir}. "
            "Download it from https://mmfit.github.io/ and extract it there."
        )
    return sorted([p.name for p in mmfit_dir.iterdir() if p.is_dir()])


def load_modality(session: str, modality: str, mmfit_dir: Path = MMFIT_DIR) -> np.ndarray | None:
    """
    Load a single sensor modality file for a session, e.g. modality=
    'sw_r_acc' or 'sw_r_gyr'. Returns None if the file is missing (not
    every participant wore every device).
    Returned array has shape (N, 5): (Frame, Timestamp, X, Y, Z).
    """
    filepath = mmfit_dir / session / f"{session}_{modality}.npy"
    if not filepath.exists():
        return None
    return np.load(filepath)


def load_labels(session: str, mmfit_dir: Path = MMFIT_DIR) -> pd.DataFrame:
    """
    Load the label CSV for a session and return it as a DataFrame with
    columns: start_frame, end_frame, reps, activity.
    """
    filepath = mmfit_dir / session / f"{session}_labels.csv"
    labels = pd.read_csv(
        filepath, header=None,
        names=["start_frame", "end_frame", "reps", "activity"]
    )
    return labels


def modality_to_dataframe(modality_array: np.ndarray, sensor_name: str) -> pd.DataFrame:
    """
    Convert a raw (N, 5) modality array [frame, timestamp, x, y, z] into a
    tidy DataFrame with named columns, e.g. accel_x, accel_y, accel_z.
    """
    cols = ["frame", "timestamp", f"{sensor_name}_x", f"{sensor_name}_y", f"{sensor_name}_z"]
    return pd.DataFrame(modality_array, columns=cols)


def load_session(session: str, modalities=("sw_r_acc", "sw_r_gyr"),
                  mmfit_dir: Path = MMFIT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge requested modalities for one session on their shared
    'frame' column, and load the corresponding labels.

    Returns:
        sensor_df: merged sensor readings, one row per frame
        labels_df: exercise set labels (start_frame, end_frame, reps, activity)
    """
    merged = None
    for modality in modalities:
        arr = load_modality(session, modality, mmfit_dir)
        if arr is None:
            print(f"[{session}] Warning: modality '{modality}' not found, skipping.")
            continue
        sensor_name = "accel" if "acc" in modality else "gyro"
        df = modality_to_dataframe(arr, sensor_name)
        # Drop the per-modality timestamp before merging so we don't get
        # duplicate/conflicting timestamp columns; frame is the shared key.
        merge_cols = ["frame"] if merged is None else ["frame"]
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df.drop(columns=["timestamp"]), on="frame", how="inner")

    if merged is None:
        raise ValueError(f"No requested modalities were found for session {session}.")

    labels_df = load_labels(session, mmfit_dir)
    return merged, labels_df


def label_frames(sensor_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign an 'activity' label to every frame in sensor_df based on which
    labeled exercise interval (if any) it falls inside. Frames outside all
    intervals are labeled 'non_activity'.
    """
    sensor_df = sensor_df.copy()
    sensor_df["activity"] = "non_activity"
    for _, row in labels_df.iterrows():
        mask = (sensor_df["frame"] >= row["start_frame"]) & (sensor_df["frame"] <= row["end_frame"])
        sensor_df.loc[mask, "activity"] = row["activity"]
    return sensor_df


def load_all_sessions(modalities=("sw_r_acc", "sw_r_gyr"),
                       mmfit_dir: Path = MMFIT_DIR) -> pd.DataFrame:
    """
    Convenience function: load every session, label every frame, tag rows
    with their session id, and concatenate into one big DataFrame.
    Use this as the entry point for the preprocessing stage.
    """
    all_dfs = []
    for session in list_sessions(mmfit_dir):
        try:
            sensor_df, labels_df = load_session(session, modalities, mmfit_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {session}: {e}")
            continue
        labeled_df = label_frames(sensor_df, labels_df)
        labeled_df["session"] = session
        all_dfs.append(labeled_df)

    if not all_dfs:
        raise RuntimeError(
            "No sessions could be loaded. Check that the MM-Fit dataset "
            "is downloaded and extracted correctly under data/raw/mm-fit/."
        )
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    # Quick smoke test: run `python src/data_loader.py` after placing the
    # dataset to confirm everything loads correctly.
    sessions = list_sessions()
    print(f"Found {len(sessions)} sessions: {sessions}")

    df = load_all_sessions()
    print(f"\nCombined shape: {df.shape}")
    print(f"\nActivity distribution:\n{df['activity'].value_counts()}")
    print(f"\nExpected activities: {MMFIT_ACTIVITIES}")
