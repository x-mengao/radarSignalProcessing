# This script hosts common signal processing functions for analysis
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def compute_fft(signal, fs):
    """
    Compute the FFT of a signal and return the frequencies, magnitude and peaks.
    
    Parameters:
    signal (numpy array): Input signal in time domain.
    fs (float): Sampling frequency.

    Returns:
    frequencies (numpy array): Frequencies corresponding to the FFT.
    X_magnitude (numpy array): Magnitude of the FFT.
    peaks (numpy array): Indices of the peaks in the FFT magnitude.
    
    # Note: following are common confusion to do fftshift after, just for clarity. However, it's not necessary, 
    # as fftfreq has already shifted the zero frequency component to the center
    # frequencies = fftshift(frequencies)  # Shift zero frequency component to center, so it ranges from -fs/2 to fs/2-1bin
    # plt.plot(frequencies, fftshift(X_magnitude)) # Shift FFT output so it aligns with shifted frequencies axis
    """
    # Compute FFT of the signal
    N = len(signal)
    X = fft(signal)
    X_magnitude = np.abs(X) / N  # Normalize the magnitude
    frequencies = fftfreq(N, 1/fs)
    # Identify peaks in the spectrum
    peaks, _ = find_peaks(X_magnitude, height=0.1)  
    return frequencies, X_magnitude, peaks