import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


DATA_PATH = "data/processed/ml_dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df[
        [
            "peak_frequency",
            "power",
            "variance",
            "spectral_energy",
            "bandwidth",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", probability=True)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    os.makedirs(REPORT_DIR, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    disp.plot(xticks_rotation=45)
    plt.title("RF Signal Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, "models/rf_classifier.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model saved successfully.")
    print("Saved: models/rf_classifier.pkl")
    print("Saved: models/scaler.pkl")


if __name__ == "__main__":
    train_model()