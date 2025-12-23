"""
main.py

FastAPI backend for ML Model Monitoring System.
Provides APIs to access alerts, metrics, drift status, and retraining decisions.
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

import json
import os
from fastapi import FastAPI
from typing import List, Dict, Any


# =========================
# 2. INITIALIZE APP
# =========================

app = FastAPI(
    title="ML Model Monitoring API",
    description="Backend API for model monitoring, drift detection, and alerting",
    version="1.0.0"
)


# =========================
# 3. FILE PATH CONFIGURATION
# =========================

PATHS = {
    "alerts": "reports/alerts/alerts_log.json",
    "baseline_metrics": "reports/baseline/performance_metrics.json",
    "latest_metrics": "reports/monitoring/latest_performance.json",
    "performance_history": "reports/monitoring/performance_history.json",
    "data_drift": "reports/drift/data_drift_report.json",
    "concept_drift": "reports/monitoring/concept_drift_report.json",
    "retrain_status": "reports/retraining/retrain_decisions.json"
}


# =========================
# 4. HELPER FUNCTION
# =========================

def load_json(path: str):
    """
    Safely loads JSON data from disk.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


# =========================
# 5. HEALTH CHECK
# =========================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ml-model-monitoring-api"
    }


# =========================
# 6. ALERTS ENDPOINT
# =========================

@app.get("/alerts")
def get_alerts():
    data = load_json(PATHS["alerts"])
    return data if data else []


# =========================
# 7. METRICS ENDPOINTS
# =========================

@app.get("/metrics/baseline")
def get_baseline_metrics():
    return load_json(PATHS["baseline_metrics"])


@app.get("/metrics/latest")
def get_latest_metrics():
    return load_json(PATHS["latest_metrics"])


@app.get("/metrics/history")
def get_performance_history():
    return load_json(PATHS["performance_history"])


# =========================
# 8. DRIFT ENDPOINTS
# =========================

@app.get("/drift/data")
def get_data_drift():
    return load_json(PATHS["data_drift"])


@app.get("/drift/concept")
def get_concept_drift():
    return load_json(PATHS["concept_drift"])


# =========================
# 9. RETRAINING STATUS
# =========================

@app.get("/retraining/status")
def get_retraining_status():
    return load_json(PATHS["retrain_status"])
