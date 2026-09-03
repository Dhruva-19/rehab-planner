"""
Feature Engineering for IMU Exercise Windows
----------------------------------------------
Converts raw (600, 6) IMU windows into a flat feature vector suitable for
tree-based models (Random Forest / XGBoost).

Input : data/processed/labeled_windows.npz
          X        -> (N, 600, 6) raw signal windows
          y        -> (N,) integer labels
          sessions -> (N,) session ids (needed later for session-based split)

Output: data/processed/features.csv
          One row per window: 77 feature columns + 'label' + 'session'
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

CHANNEL_NAMES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
SAMPLING_RATE = 200  # Hz, confirmed empirically in Day 3
EPS = 1e-10          # small constant to avoid log(0) / divide-by-zero


def compute_time_domain_features(signal):
    """
    Compute time-domain statistical features for a single 1D signal.

    signal: 1D numpy array, shape (600,) -- one channel of one window
    Returns: dict of {feature_name: value}
    """
    signal = np.asarray(signal, dtype=np.float64)

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    range_val = max_val - min_val
    rms_val = np.sqrt(np.mean(signal ** 2))

    # skew/kurtosis are undefined (return nan) for a constant (zero-variance)
    # signal -- guard against that explicitly rather than relying on scipy.
    if std_val < EPS:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = stats.skew(signal)
        kurt_val = stats.kurtosis(signal)

    # Zero-crossing rate: how often the signal changes sign, normalized
    # by signal length. Higher = more oscillatory motion.
    signs = np.sign(signal)
    signs[signs == 0] = 1  # treat exact zeros as positive so diff works cleanly
    zero_crossings = np.sum(np.diff(signs) != 0)
    zcr_val = zero_crossings / len(signal)

    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'rms': rms_val,
        'skew': skew_val,
        'kurtosis': kurt_val,
        'zcr': zcr_val,
    }


def compute_freq_domain_features(signal, fs=SAMPLING_RATE):
    """
    Compute frequency-domain features via FFT for a single 1D signal.

    signal: 1D numpy array, shape (600,)
    Returns: dict with dominant_freq, spectral_energy, spectral_entropy
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    fft_vals = rfft(signal)
    fft_freqs = rfftfreq(n, d=1.0 / fs)
    power_spectrum = np.abs(fft_vals) ** 2

    # Skip index 0 -- that's the DC component (the signal's mean), not a
    # real oscillation frequency, and would always dominate.
    if len(power_spectrum) > 1:
        dominant_freq = fft_freqs[np.argmax(power_spectrum[1:]) + 1]
    else:
        dominant_freq = 0.0

    spectral_energy = np.sum(power_spectrum)

    # Spectral entropy: normalize power spectrum into a probability
    # distribution, then compute Shannon entropy. Low entropy = energy
    # concentrated at a few frequencies (smooth, rhythmic motion).
    # High entropy = energy spread out (chaotic / noisy motion).
    total_power = np.sum(power_spectrum) + EPS
    p = power_spectrum / total_power
    spectral_entropy = -np.sum(p * np.log2(p + EPS))

    return {
        'dominant_freq': dominant_freq,
        'spectral_energy': spectral_energy,
        'spectral_entropy': spectral_entropy,
    }


def compute_cross_channel_features(window):
    """
    Compute features that combine multiple channels within a window.

    window: 2D numpy array, shape (600, 6)
    Returns: dict with sma_acc, sma_gyro, and 3 accelerometer axis-pair
             correlations.
    """
    window = np.asarray(window, dtype=np.float64)

    acc = window[:, 0:3]   # acc_x, acc_y, acc_z
    gyro = window[:, 3:6]  # gyro_x, gyro_y, gyro_z

    # Signal Magnitude Area: average of summed absolute values across
    # axes, per timestep. Captures overall movement intensity in an
    # orientation-independent way.
    sma_acc = np.mean(np.sum(np.abs(acc), axis=1))
    sma_gyro = np.mean(np.sum(np.abs(gyro), axis=1))

    def safe_corr(a, b):
        # If either axis is constant, correlation is undefined -> return 0
        if np.std(a) < EPS or np.std(b) < EPS:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    corr_acc_xy = safe_corr(acc[:, 0], acc[:, 1])
    corr_acc_yz = safe_corr(acc[:, 1], acc[:, 2])
    corr_acc_xz = safe_corr(acc[:, 0], acc[:, 2])

    return {
        'sma_acc': sma_acc,
        'sma_gyro': sma_gyro,
        'corr_acc_xy': corr_acc_xy,
        'corr_acc_yz': corr_acc_yz,
        'corr_acc_xz': corr_acc_xz,
    }


def extract_features_from_window(window):
    """
    Extract the full 77-feature set from a single window.

    window: 2D numpy array, shape (600, 6)
    Returns: dict of ALL features, keys prefixed by channel name
             e.g. 'acc_x_mean', 'acc_x_std', ..., 'gyro_z_spectral_entropy',
             plus the 5 cross-channel keys.
    """
    features = {}

    for i, channel_name in enumerate(CHANNEL_NAMES):
        signal = window[:, i]

        time_feats = compute_time_domain_features(signal)
        freq_feats = compute_freq_domain_features(signal)

        for feat_name, value in {**time_feats, **freq_feats}.items():
            features[f'{channel_name}_{feat_name}'] = value

    cross_feats = compute_cross_channel_features(window)
    features.update(cross_feats)

    return features


def build_feature_matrix(X, y, sessions):
    """
    Apply feature extraction to every window in the dataset.

    X: shape (N, 600, 6)
    y: shape (N,)
    sessions: shape (N,)
    Returns: pandas DataFrame, one row per window
    """
    n_windows = X.shape[0]
    rows = []

    for idx in range(n_windows):
        feats = extract_features_from_window(X[idx])
        rows.append(feats)

        if (idx + 1) % 1000 == 0 or (idx + 1) == n_windows:
            print(f"  processed {idx + 1}/{n_windows} windows")

    features_df = pd.DataFrame(rows)

    # Final safety net: replace any stray NaN/inf that slipped through
    # (e.g. from an unexpected zero-variance edge case) with 0.
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features_df['label'] = y
    features_df['session'] = sessions

    return features_df


if __name__ == "__main__":
    print("Loading labeled windows...")
    data = np.load("data/processed/labeled_windows.npz", allow_pickle=True)
    X, y, sessions = data['X'], data['y'], data['sessions']
    print(f"  X shape: {X.shape}, y shape: {y.shape}, sessions shape: {sessions.shape}")

    print("Extracting features...")
    features_df = build_feature_matrix(X, y, sessions)

    print(f"\nFinal feature matrix shape: {features_df.shape}")
    print(f"Expected: ({X.shape[0]}, 79)  <- 77 features + label + session")
    print(features_df.head())

    out_path = "data/processed/features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
