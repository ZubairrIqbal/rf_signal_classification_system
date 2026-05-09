import sys
import os
import time
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from features import extract_features


MODEL_PATH = "models/rf_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"


app = FastAPI(
    title="RF Signal Classification API",
    description="DSP + ML based RF signal classification system",
    version="1.0.0"
)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class SignalInput(BaseModel):
    signal: List[float]


@app.get("/")
def root():
    return {
        "message": "RF Signal Classification API is running",
        "status": "active"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict")
def predict(input_data: SignalInput):
    try:
        start_time = time.time()

        signal = np.array(input_data.signal)

        if len(signal) < 100:
            raise HTTPException(
                status_code=400,
                detail="Signal length must be at least 100 samples"
            )

        features = extract_features(signal)
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))

        latency_ms = (time.time() - start_time) * 1000

        return {
            "prediction": prediction,
            "confidence": confidence,
            "features": {
                "peak_frequency": float(features[0]),
                "power": float(features[1]),
                "variance": float(features[2]),
                "spectral_energy": float(features[3]),
                "bandwidth": float(features[4])
            },
            "latency_ms": round(latency_ms, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))