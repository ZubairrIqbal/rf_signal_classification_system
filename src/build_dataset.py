import os
import numpy as np
import pandas as pd

from features import extract_features


DATA_DIR = "data/raw"
OUTPUT_CSV = "data/processed/ml_dataset.csv"


def build_feature_dataset():

    dataset = []

    for class_name in os.listdir(DATA_DIR):

        class_path = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            if not file_name.endswith(".npy"):
                continue

            file_path = os.path.join(class_path, file_name)

            signal = np.load(file_path)

            features = extract_features(signal)

            row = {
                "peak_frequency": features[0],
                "power": features[1],
                "variance": features[2],
                "spectral_energy": features[3],
                "bandwidth": features[4],
                "label": class_name
            }

            dataset.append(row)

    df = pd.DataFrame(dataset)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print("ML Dataset Created Successfully")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)


if __name__ == "__main__":
    build_feature_dataset()