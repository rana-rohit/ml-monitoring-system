"""
alert_engine.py

This module acts as the decision layer of the monitoring system.
It reads monitoring outputs and generates alerts based on predefined rules.
"""

# =========================
# 1. IMPORT REQUIRED LIBRARIES
# =========================

import json
import os
from datetime import datetime


# =========================
# 2. FILE PATH CONFIGURATION
# =========================

BASELINE_METRICS_PATH = "reports/baseline/performance_metrics.json"
DATA_DRIFT_PATH = "reports/drift/data_drift_report.json"
CONCEPT_DRIFT_PATH = "reports/monitoring/concept_drift_report.json"
PERFORMANCE_HISTORY_PATH = "reports/monitoring/performance_history.json"

ALERTS_DIR = "reports/alerts"
ALERTS_FILE = os.path.join(ALERTS_DIR, "alerts_log.json")

os.makedirs(ALERTS_DIR, exist_ok=True)


# =========================
# 3. HELPER FUNCTION TO LOAD JSON SAFELY
# =========================

def load_json(path):
    """
    Safely loads a JSON file.
    Returns None if file does not exist.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


# =========================
# 4. ALERT GENERATION LOGIC
# =========================

alerts = []


def create_alert(level, message, source):
    """
    Creates a standardized alert dictionary.
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "source": source,
        "message": message
    }


# =========================
# 5. DATA DRIFT ALERTS
# =========================

data_drift_report = load_json(DATA_DRIFT_PATH)

if data_drift_report:
    drifted_features = [
        feature for feature, stats in data_drift_report.items()
        if stats.get("drift_detected", False)
    ]

    if drifted_features:
        alerts.append(
            create_alert(
                level="WARNING",
                source="data_drift",
                message=f"Data drift detected in {len(drifted_features)} features."
            )
        )
    else:
        alerts.append(
            create_alert(
                level="INFO",
                source="data_drift",
                message="No significant data drift detected."
            )
        )


# =========================
# 6. CONCEPT DRIFT ALERTS
# =========================

concept_drift_report = load_json(CONCEPT_DRIFT_PATH)

if concept_drift_report:
    if concept_drift_report.get("concept_drift_detected", False):
        alerts.append(
            create_alert(
                level="WARNING",
                source="concept_drift",
                message="Concept drift detected based on prediction distribution."
            )
        )
    else:
        alerts.append(
            create_alert(
                level="INFO",
                source="concept_drift",
                message="No concept drift detected."
            )
        )


# =========================
# 7. PERFORMANCE DEGRADATION ALERTS
# =========================

performance_history = load_json(PERFORMANCE_HISTORY_PATH)

if performance_history:
    degraded_batches = [
        batch for batch in performance_history
        if batch.get("performance_degraded", False)
    ]

    if degraded_batches:
        alerts.append(
            create_alert(
                level="CRITICAL",
                source="performance_monitor",
                message=f"Performance degraded in {len(degraded_batches)} batches."
            )
        )
    else:
        alerts.append(
            create_alert(
                level="INFO",
                source="performance_monitor",
                message="Model performance within acceptable range."
            )
        )


# =========================
# 8. SAVE ALERTS LOG
# =========================

# Load existing alerts if present
existing_alerts = load_json(ALERTS_FILE)
if existing_alerts is None:
    existing_alerts = []

# Append new alerts
existing_alerts.extend(alerts)

# Save back to disk
with open(ALERTS_FILE, "w") as f:
    json.dump(existing_alerts, f, indent=4)


# =========================
# 9. FINAL OUTPUT
# =========================

print("🚨 Alerting engine run complete.")
print(f"📣 Alerts generated this run: {len(alerts)}")
print("📝 Alerts log saved to:", ALERTS_FILE)

for alert in alerts:
    print(f"[{alert['level']}] {alert['source']} → {alert['message']}")
