"""
performance_monitor.py

This module monitors model performance over time by:
1. Simulating multiple production batches
2. Computing performance metrics for each batch
3. Storing metrics with timestamps
4. Detecting performance degradation
"""

# =========================
# 1. IMPORT REQUIRED LIBRARIES
# =========================

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib


# =========================
# 2. LOAD TRAINED MODEL
# =========================

MODEL_PATH = "models/baseline/model.joblib"
model = joblib.load(MODEL_PATH)


# =========================
# 3. LOAD DATA
# =========================

dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

# Explanation:
# In real production, this data would arrive in time-based batches.


# =========================
# 4. PERFORMANCE MONITORING CONFIG
# =========================

OUTPUT_DIR = "reports/monitoring"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_FILE = os.path.join(OUTPUT_DIR, "performance_history.json")

BATCH_SIZE = 50
DEGRADATION_THRESHOLD = 0.90  # 90% of baseline accuracy


# =========================
# 5. LOAD BASELINE PERFORMANCE
# =========================

with open("reports/baseline/performance_metrics.json", "r") as f:
    baseline_metrics = json.load(f)

baseline_accuracy = baseline_metrics["accuracy"]


# =========================
# 6. SIMULATE STREAMING BATCHES
# =========================

performance_history = []

for i in range(0, len(X), BATCH_SIZE):
    X_batch = X.iloc[i:i + BATCH_SIZE]
    y_batch = y.iloc[i:i + BATCH_SIZE]

    # skip incomplete batches
    if len(X_batch) < BATCH_SIZE:
        continue

    # model predictions
    y_pred = model.predict(X_batch)
    y_pred_proba = model.predict_proba(X_batch)[:, 1]

    # compute metrics
    batch_metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy": float(accuracy_score(y_batch, y_pred)),
        "precision": float(precision_score(y_batch, y_pred)),
        "recall": float(recall_score(y_batch, y_pred)),
        "roc_auc": float(roc_auc_score(y_batch, y_pred_proba))
    }

    # performance degradation check
    batch_metrics["performance_degraded"] = bool(
        batch_metrics["accuracy"] < DEGRADATION_THRESHOLD * baseline_accuracy
    )

    performance_history.append(batch_metrics)


# =========================
# 7. SAVE PERFORMANCE HISTORY
# =========================

with open(METRICS_FILE, "w") as f:
    json.dump(performance_history, f, indent=4)


# =========================
# 8. FINAL OUTPUT
# =========================

degraded_batches = [
    m for m in performance_history if m["performance_degraded"]
]

print("📉 Performance monitoring complete.")
print(f"📊 Total batches monitored: {len(performance_history)}")
print(f"⚠️ Degraded batches: {len(degraded_batches)}")
print("📝 Performance history saved to:", METRICS_FILE)
