"""
concept_drift.py

This module detects concept drift by monitoring changes in
model prediction distributions over time.

We compare:
- baseline prediction probabilities
- current (production) prediction probabilities
"""

# =========================
# 1. IMPORT REQUIRED LIBRARIES
# =========================

import json
import os

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
import joblib


# =========================
# 2. LOAD TRAINED MODEL
# =========================

MODEL_PATH = "models/baseline/model.joblib"
model = joblib.load(MODEL_PATH)

# Explanation:
# Concept drift monitoring uses the deployed model.


# =========================
# 3. LOAD DATA
# =========================

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

# simulate baseline and production splits
X_baseline = X.sample(frac=0.5, random_state=42)
X_production = X.sample(frac=0.5, random_state=99)

# Explanation:
# Baseline ≈ training-era behavior
# Production ≈ current behavior


# =========================
# 4. GENERATE PREDICTION PROBABILITIES
# =========================

baseline_probs = model.predict_proba(X_baseline)[:, 1]
production_probs = model.predict_proba(X_production)[:, 1]

# We focus on probabilities because they contain richer information
# than hard class labels.


# =========================
# 5. CONCEPT DRIFT DETECTION
# =========================

ks_statistic, p_value = ks_2samp(baseline_probs, production_probs)

P_VALUE_THRESHOLD = 0.05

concept_drift_detected = bool(p_value < P_VALUE_THRESHOLD)

concept_drift_report = {
    "ks_statistic": float(ks_statistic),
    "p_value": float(p_value),
    "concept_drift_detected": concept_drift_detected
}


# =========================
# 6. SAVE CONCEPT DRIFT REPORT
# =========================

OUTPUT_DIR = "reports/monitoring"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "concept_drift_report.json")

with open(output_path, "w") as f:
    json.dump(concept_drift_report, f, indent=4)


# =========================
# 7. FINAL OUTPUT
# =========================

print("🧠 Concept drift analysis complete.")

if concept_drift_detected:
    print("⚠️ Concept drift detected!")
else:
    print("✅ No significant concept drift detected.")

print("📝 Report saved to:", output_path)
