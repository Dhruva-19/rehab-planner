"""
Day 2 - Step 2: Sampling Rate Detection (phone-only: sp_r_acc, sp_r_gyro)
---------------------------------------------------------------------------
Goal: measure the ACTUAL sampling rate and jitter of the phone's accelerometer
and gyroscope streams directly from their timestamp columns, rather than
trusting a nominal "100Hz" figure. Also check whether acc and gyro are
time-aligned (same start/end, same effective rate) since we'll need to
merge them onto a common time grid before windowing.
 
Recall from Day 1: each .npy file has shape (N, 5) with columns:
    [Frame, Timestamp, X, Y, Z]
Timestamp is assumed to be in seconds (verified below - if it looks like
milliseconds instead, the script will flag it).
 
Run from your project root.
"""
 
import numpy as np
from pathlib import Path
 
DATA_ROOT = Path("data/raw/mm-fit")
SESSIONS = [f"w{i:02d}" for i in range(21)]  # w00 .. w20
MODALITIES = ["sp_r_acc", "sp_r_gyr"]  # confirmed filenames: MM-Fit uses "gyr" not "gyro"
 
TIMESTAMP_COL = 1  # column index confirmed from Day 1 (Frame, Timestamp, X, Y, Z)
 
 
def normalize_timestamps(ts: np.ndarray) -> np.ndarray:
    """
    Detect whether timestamps are in milliseconds or seconds, and convert to seconds.
 
    Heuristic: for a gym session, the median gap between consecutive samples
    should be well under 1 second (since sensors sample many times per second).
    If the raw median gap is >> 1, the timestamps are almost certainly in
    milliseconds (or similar), so we divide by 1000.
    """
    raw_dt = np.median(np.diff(ts))
    if raw_dt > 1.0:  # gap bigger than 1 "unit" is implausible for a sensor stream
        return ts / 1000.0
    return ts
 
 
def load_modality(session: str, modality: str) -> np.ndarray | None:
    """Load one modality file for one session. Returns None if missing."""
    path = DATA_ROOT / session / f"{session}_{modality}.npy"
    if not path.exists():
        return None
    return np.load(path)
 
 
def analyze_timestamps(timestamps: np.ndarray) -> dict:
    """Compute effective sampling rate, jitter, and gap stats for a timestamp array."""
    dt = np.diff(timestamps)
    dt = dt[dt > 0]  # guard against any duplicate/out-of-order timestamps
 
    median_dt = np.median(dt)
    effective_hz = 1.0 / median_dt if median_dt > 0 else float("nan")
    jitter_std_ms = np.std(dt) * 1000
    max_gap_ms = np.max(dt) * 1000
    # Flag gaps that are more than 3x the median interval (likely dropped samples)
    gap_threshold = median_dt * 3
    n_large_gaps = int(np.sum(dt > gap_threshold))
 
    return {
        "n_samples": len(timestamps),
        "duration_s": timestamps[-1] - timestamps[0],
        "effective_hz": effective_hz,
        "jitter_std_ms": jitter_std_ms,
        "max_gap_ms": max_gap_ms,
        "n_large_gaps": n_large_gaps,
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
    }
 
 
def main():
    print(f"{'Session':<8} {'Modality':<12} {'Hz':<8} {'Jitter(ms)':<12} "
          f"{'MaxGap(ms)':<12} {'#Gaps':<7} {'Samples':<9} {'Duration(s)'}")
    print("-" * 90)
 
    summary = {mod: [] for mod in MODALITIES}
 
    for session in SESSIONS:
        stats_per_modality = {}
        for modality in MODALITIES:
            arr = load_modality(session, modality)
            if arr is None:
                print(f"{session:<8} {modality:<12} MISSING")
                continue
 
            ts = normalize_timestamps(arr[:, TIMESTAMP_COL])
            stats = analyze_timestamps(ts)
            stats_per_modality[modality] = stats
            summary[modality].append(stats["effective_hz"])
 
            print(f"{session:<8} {modality:<12} {stats['effective_hz']:<8.2f} "
                  f"{stats['jitter_std_ms']:<12.3f} {stats['max_gap_ms']:<12.1f} "
                  f"{stats['n_large_gaps']:<7} {stats['n_samples']:<9} "
                  f"{stats['duration_s']:.2f}")
 
        # Check acc/gyro time alignment for this session
        if "sp_r_acc" in stats_per_modality and "sp_r_gyro" in stats_per_modality:
            acc_s = stats_per_modality["sp_r_acc"]
            gyro_s = stats_per_modality["sp_r_gyro"]
            start_diff_ms = abs(acc_s["start_time"] - gyro_s["start_time"]) * 1000
            end_diff_ms = abs(acc_s["end_time"] - gyro_s["end_time"]) * 1000
            if start_diff_ms > 50 or end_diff_ms > 50:  # >50ms mismatch flagged
                print(f"  -> ALIGNMENT WARNING: start diff {start_diff_ms:.1f}ms, "
                      f"end diff {end_diff_ms:.1f}ms")
 
    print("\n" + "=" * 40)
    print("SUMMARY ACROSS ALL SESSIONS")
    print("=" * 40)
    for modality, rates in summary.items():
        if rates:
            print(f"{modality}: mean={np.mean(rates):.2f}Hz, "
                  f"min={np.min(rates):.2f}Hz, max={np.max(rates):.2f}Hz, "
                  f"std={np.std(rates):.3f}Hz")
 
 
if __name__ == "__main__":
    main()
 