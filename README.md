Pump Predictive Maintenance

This project builds a machine learning based predictive maintenance system for centrifugal pumps. The model analyzes vibration, flow-rate and pressure patterns to detect early signs of degradation, helping avoid unexpected failures and improving maintenance scheduling.

The workflow includes data preprocessing, feature extraction, anomaly detection and supervised ML modeling. Results include performance metrics, feature insights and visual degradation trends for real-world interpretability.

Key Features

Cleans and preprocesses raw sensor data

Extracts important time-domain and frequency-domain features

Detects abnormal behavior using statistical and ML-based methods

Trains predictive models such as Random Forest and XGBoost

Generates visual diagnostics for pump health monitoring

Saves trained models and results for deployment

Project Structure
pump-predictive-maintenance/
│
├── data/
│   ├── raw/               # Raw vibration/pressure/flow datasets
│   ├── processed/         # Cleaned and feature-ready data
│   └── README.md
│
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── feature_engineering.ipynb # Feature extraction
│   └── model_training.ipynb      # ML experiments
│
├── src/
│   ├── data_preprocessing.py     # Cleaning, smoothing, normalization
│   ├── feature_extraction.py     # Domain features
│   ├── train_model.py            # Model training pipeline
│   └── evaluate_model.py         # Performance evaluation
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

Installation

Make sure Python 3.8+ is installed.

pip install -r requirements.txt

Usage
1. Train the Model
python src/train_model.py

2. Evaluate the Model
python src/evaluate_model.py

3. Explore Notebooks

Open the notebooks in notebooks/ for:

Exploratory data analysis

Feature engineering

Model experimentation

Results

All generated outputs are stored in the results/ directory:

Model metrics

Confusion matrix

Feature importance

Degradation trend visualizations

Saved trained model

Technologies Used

Python

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

Jupyter

License

This project is licensed under the MIT License.
