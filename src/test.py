import numpy as np
import matplotlib.pyplot as plt

from dsp import preprocess_signal


signal = np.load("data/raw/low_freq_signal/low_freq_signal_0.npy")

processed = preprocess_signal(signal)

plt.figure(figsize=(10, 4))
plt.plot(signal[:300], label="Raw Signal")
plt.plot(processed[:300], label="Processed Signal")
plt.legend()
plt.title("Raw vs Processed Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()
import numpy as np

from features import extract_features


signal = np.load("data/raw/low_freq_signal/low_freq_signal_0.npy")

features = extract_features(signal)

print("Extracted Features:")
print(features)