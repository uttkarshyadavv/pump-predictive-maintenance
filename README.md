# Pump Predictive Maintenance

A complete machine learning project for predicting failures in centrifugal pumps using vibration, pressure and flow-rate data. The system performs preprocessing, feature extraction, anomaly detection and supervised learning to identify early degradation and support proactive maintenance decisions.

---

## Features
- Cleans and preprocesses raw sensor data
- Extracts time-domain and frequency-domain features
- Detects abnormal operating patterns
- Trains ML models (Random Forest, XGBoost etc.)
- Visualizes pump health and degradation trends
- Saves results and trained models for deployment

---

## Project Structure
pump-predictive-maintenance/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── model/
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore

---

## Installation
```bash
pip install -r requirements.txt
