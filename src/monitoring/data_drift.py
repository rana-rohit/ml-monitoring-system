"""
data_drift.py

This module detects data drift by comparing:
- training-time feature distributions
- production-time feature distributions

We use the Kolmogorov–Smirnov (KS) test for numerical features.
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


# =========================
# 2. LOAD BASELINE FEATURE STATISTICS
# =========================

BASELINE_STATS_PATH = "reports/baseline/feature_stats.json"

with open(BASELINE_STATS_PATH, "r") as f:
    baseline_feature_stats = json.load(f)

# Explanation:
# These statistics describe what the data looked like at training time.


# =========================
# 3. LOAD BASELINE TRAINING DATA (REFERENCE DISTRIBUTION)
# =========================

# load the original dataset again
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# simulate "production data" by sampling a subset
X_prod = X.sample(frac=0.2, random_state=99)

# Explanation:
# In real systems, X_prod would come from live traffic.


# =========================
# 4. DATA DRIFT DETECTION LOGIC
# =========================

DRIFT_RESULTS_DIR = "reports/drift"
os.makedirs(DRIFT_RESULTS_DIR, exist_ok=True)

drift_report = {}

# threshold for drift detection
P_VALUE_THRESHOLD = 0.05

for feature in X.columns:
    # reference (training-like) data
    reference_data = X[feature]

    # production data
    production_data = X_prod[feature]

    # perform KS test
    ks_statistic, p_value = ks_2samp(reference_data, production_data)

    # determine if drift is detected
    drift_detected = bool(p_value < P_VALUE_THRESHOLD)

    drift_report[feature] = {
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < P_VALUE_THRESHOLD)
    }

# Explanation:
# - ks_statistic measures distance between distributions
# - p_value indicates statistical significance
# - drift_detected is a boolean flag


# =========================
# 5. SAVE DRIFT REPORT
# =========================

output_path = os.path.join(DRIFT_RESULTS_DIR, "data_drift_report.json")

with open(output_path, "w") as f:
    json.dump(drift_report, f, indent=4)


# =========================
# 6. SUMMARY OUTPUT
# =========================

drifted_features = [
    feature for feature, stats in drift_report.items()
    if stats["drift_detected"]
]

print("🔍 Data drift detection complete.")
print(f"📊 Drifted features count: {len(drifted_features)}")

if drifted_features:
    print("⚠️ Drift detected in the following features:")
    for feature in drifted_features:
        print(" -", feature)
else:
    print("✅ No significant data drift detected.")

print("📝 Drift report saved to:", output_path)