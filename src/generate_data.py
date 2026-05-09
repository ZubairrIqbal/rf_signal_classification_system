import os
import numpy as np
import pandas as pd

# -------------------------
# Configuration
# -------------------------
FS = 1000                 # sample rate: 1000 samples/sec
DURATION = 1              # 1 second signal
N_SAMPLES = FS * DURATION

OUTPUT_DIR = "data/raw"
CSV_PATH = "data/processed/features_labels.csv"

CLASSES = {
    "low_freq_signal": 50,
    "mid_freq_signal": 120,
    "high_freq_signal": 250,
    "interference_signal": 350
}

SAMPLES_PER_CLASS = 300


# -------------------------
# Generate one RF-like signal
# -------------------------
def generate_signal(freq, fs, duration, noise_level=0.2):
    t = np.arange(0, duration, 1 / fs)

    amplitude = np.random.uniform(0.7, 1.3)
    phase = np.random.uniform(0, 2 * np.pi)
    freq_shift = np.random.uniform(-3, 3)

    signal = amplitude * np.sin(2 * np.pi * (freq + freq_shift) * t + phase)

    noise = np.random.normal(0, noise_level, len(t))

    rf_signal = signal + noise

    return rf_signal


# -------------------------
# Main dataset generation
# -------------------------
def create_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata = []

    sample_id = 0

    for class_name, freq in CLASSES.items():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            noise_level = np.random.uniform(0.1, 0.6)

            signal = generate_signal(
                freq=freq,
                fs=FS,
                duration=DURATION,
                noise_level=noise_level
            )

            file_name = f"{class_name}_{i}.npy"
            file_path = os.path.join(class_dir, file_name)

            np.save(file_path, signal)

            metadata.append({
                "sample_id": sample_id,
                "file_path": file_path,
                "label": class_name,
                "base_frequency": freq,
                "noise_level": noise_level
            })

            sample_id += 1

    metadata_df = pd.DataFrame(metadata)

    os.makedirs("data/processed", exist_ok=True)
    metadata_df.to_csv("data/processed/metadata.csv", index=False)

    print("Dataset created successfully.")
    print(f"Total samples: {len(metadata_df)}")
    print(metadata_df.head())


if __name__ == "__main__":
    create_dataset()