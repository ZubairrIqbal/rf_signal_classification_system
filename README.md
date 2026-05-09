# Intelligent Signal Processing & ML-Based RF Classification System

This project implements an end-to-end RF signal classification pipeline using Python, DSP, machine learning, and FastAPI.

The system generates synthetic RF-like signals, applies DSP preprocessing, extracts FFT-based features, trains a lightweight ML classifier, and deploys the model through a FastAPI inference service.

## Pipeline

Synthetic RF Signal
→ Noise Injection
→ DSP Preprocessing
→ FFT Feature Extraction
→ ML Classification
→ FastAPI Deployment

## Key Features

- Synthetic RF signal generation
- Noise, amplitude, phase, and frequency variation
- Bandpass filtering and normalization
- FFT-based spectral feature extraction
- SVM-based classification
- Confusion matrix evaluation
- FastAPI inference API
- Prediction confidence and latency measurement

## Future Extension

The simulated signal source can be replaced with RTL-SDR or USRP hardware input using GNU Radio without changing the ML and API layers.