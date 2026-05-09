import numpy as np
from scipy.signal import butter, filtfilt


FS = 1000


def normalize_signal(signal):
    """
    Normalize signal to zero mean and unit variance.
    """
    signal = np.asarray(signal)

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal - mean

    return (signal - mean) / std


def lowpass_filter(signal, cutoff=400, fs=FS, order=4):
    """
    Remove very high frequency noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype="low")
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def bandpass_filter(signal, lowcut=20, highcut=420, fs=FS, order=4):
    """
    Keep useful RF-like signal band and remove unwanted components.
    """
    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def preprocess_signal(signal):
    """
    Full DSP preprocessing pipeline.
    """
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal)
    signal = normalize_signal(signal)

    return signal