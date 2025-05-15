import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns of a price series."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_fft(log_returns: pd.Series, sampling_rate: float = 1.0):
    """Compute the FFT and frequency spectrum of the log returns."""
    N = len(log_returns)
    freqs = fftfreq(N, d=sampling_rate)
    fft_values = fft(log_returns.values.astype(np.float64))
    freqs_half = freqs[:N // 2]
    amplitudes = np.abs(fft_values[:N // 2])
    return freqs_half, amplitudes

def detect_dominant_cycles(freqs, amplitudes, method='median_std', distance=2):
    amplitudes = np.asarray(amplitudes).flatten()
    freqs = np.asarray(freqs).flatten()

    if method == 'max_ratio':
        threshold = 0.3 * np.max(amplitudes)
    elif method == 'median_std':
        threshold = np.median(amplitudes) + 4 * np.std(amplitudes)
    elif method == 'percentile':
        threshold = np.percentile(amplitudes, 95)
    else:
        raise ValueError("Unknown method for thresholding.")

    peaks, props = find_peaks(amplitudes, height=threshold, distance=distance)
    return freqs[peaks], amplitudes[peaks]



def plot_fft_spectrum(freqs: np.ndarray, amplitudes: np.ndarray, peak_freqs: np.ndarray, peak_amps: np.ndarray):
    """Plot FFT spectrum with detected peaks."""
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, amplitudes, label="FFT Spectrum")
    plt.plot(peak_freqs, peak_amps, "x", label="Detected Peaks")

    for i, freq in enumerate(peak_freqs):
        if freq != 0:
            period = round(1 / freq, 1)
            plt.annotate(f"F={freq:.3f}\nP={period}d", (freq, peak_amps[i]), xytext=(5, 5),
                         textcoords='offset points', ha='center')

    plt.title("FFT Spectrum of Log Returns (cycles/day)")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_price_series(df: pd.DataFrame, column: str = "Close"):
    """Plot price series over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[column], label="Price")
    plt.title("Price Series")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()