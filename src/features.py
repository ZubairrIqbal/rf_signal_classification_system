import numpy as np

from dsp import preprocess_signal, FS


def compute_fft(signal, fs=FS):
    """
    Convert time-domain signal into frequency-domain spectrum.
    """
    signal = preprocess_signal(signal)

    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)

    # Keep only positive frequencies
    positive_idx = freqs >= 0
    freqs = freqs[positive_idx]
    magnitude = np.abs(fft_values[positive_idx])

    return freqs, magnitude


def extract_features(signal, fs=FS):
    """
    Extract meaningful DSP-based features from one signal.
    """
    freqs, magnitude = compute_fft(signal, fs)

    peak_index = np.argmax(magnitude)
    peak_frequency = freqs[peak_index]

    processed_signal = preprocess_signal(signal)

    power = np.mean(processed_signal ** 2)
    variance = np.var(processed_signal)
    spectral_energy = np.sum(magnitude ** 2)

    # Bandwidth estimate: frequencies above 50% of max magnitude
    threshold = 0.5 * np.max(magnitude)
    active_freqs = freqs[magnitude >= threshold]

    if len(active_freqs) > 0:
        bandwidth = active_freqs[-1] - active_freqs[0]
    else:
        bandwidth = 0

    return np.array([
        peak_frequency,
        power,
        variance,
        spectral_energy,
        bandwidth
    ])