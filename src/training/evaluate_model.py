"""
evaluate_model.py

This script simulates how a trained model behaves in production.
It:
1. Loads the trained baseline model
2. Runs predictions on unseen data
3. Computes evaluation metrics
4. Outputs results for monitoring

IMPORTANT:
No training happens here. This mimics real production inference.
"""

# =========================
# 1. IMPORT REQUIRED LIBRARIES
# =========================

import json
import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import joblib


# =========================
# 2. LOAD SAVED MODEL
# =========================

MODEL_PATH = "models/baseline/model.joblib"

# load the trained model from disk
model = joblib.load(MODEL_PATH)

# Explanation:
# In production, models are loaded, not retrained.


# =========================
# 3. LOAD DATA (SIMULATED PRODUCTION DATA)
# =========================

# load the same dataset but we will pretend this is "new" data
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

# simulate production data by taking only a slice
X_prod = X.sample(frac=0.2, random_state=7)
y_prod = y.loc[X_prod.index]

# Explanation:
# In real systems, this would come from live traffic.


# =========================
# 4. RUN MODEL INFERENCE
# =========================

# class predictions
y_pred = model.predict(X_prod)

# probability predictions
y_pred_proba = model.predict_proba(X_prod)[:, 1]


# =========================
# 5. COMPUTE PERFORMANCE METRICS
# =========================

performance_metrics = {
    "accuracy": accuracy_score(y_prod, y_pred),
    "precision": precision_score(y_prod, y_pred),
    "recall": recall_score(y_prod, y_pred),
    "roc_auc": roc_auc_score(y_prod, y_pred_proba)
}


# =========================
# 6. SAVE EVALUATION REPORT
# =========================

OUTPUT_DIR = "reports/monitoring"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "latest_performance.json")

with open(output_path, "w") as f:
    json.dump(performance_metrics, f, indent=4)


# =========================
# 7. FINAL OUTPUT
# =========================

print("📈 Model evaluation complete.")
print("📊 Current performance:", performance_metrics)
print("📝 Report saved to:", output_path)