"""
retrain_controller.py

This module decides whether model retraining should be triggered
based on alert patterns over time.
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================

import json
import os
from datetime import datetime, timedelta


# =========================
# 2. FILE PATHS
# =========================

ALERTS_FILE = "reports/alerts/alerts_log.json"
RETRAIN_LOG_DIR = "reports/retraining"
RETRAIN_LOG_FILE = os.path.join(RETRAIN_LOG_DIR, "retrain_decisions.json")

os.makedirs(RETRAIN_LOG_DIR, exist_ok=True)


# =========================
# 3. CONFIGURATION
# =========================

# retrain if CRITICAL alerts appear more than this count
CRITICAL_ALERT_THRESHOLD = 1

# look back window (hours)
LOOKBACK_HOURS = 24


# =========================
# 4. LOAD ALERTS
# =========================

def load_alerts():
    if not os.path.exists(ALERTS_FILE):
        return []

    with open(ALERTS_FILE, "r") as f:
        return json.load(f)


# =========================
# 5. RETRAIN DECISION LOGIC
# =========================

def should_retrain(alerts):
    """
    Determines whether retraining should be triggered.
    """
    now = datetime.utcnow()
    cutoff_time = now - timedelta(hours=LOOKBACK_HOURS)

    recent_critical_alerts = [
        alert for alert in alerts
        if alert["level"] == "CRITICAL"
        and datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
    ]

    return len(recent_critical_alerts) >= CRITICAL_ALERT_THRESHOLD


# =========================
# 6. MAKE DECISION
# =========================

alerts = load_alerts()
retrain_required = should_retrain(alerts)

decision = {
    "timestamp": datetime.utcnow().isoformat(),
    "retrain_required": retrain_required,
    "reason": (
        "CRITICAL performance degradation detected"
        if retrain_required
        else "System performance within acceptable limits"
    )
}


# =========================
# 7. SAVE DECISION LOG
# =========================

existing_logs = []
if os.path.exists(RETRAIN_LOG_FILE):
    with open(RETRAIN_LOG_FILE, "r") as f:
        existing_logs = json.load(f)

existing_logs.append(decision)

with open(RETRAIN_LOG_FILE, "w") as f:
    json.dump(existing_logs, f, indent=4)


# =========================
# 8. OUTPUT
# =========================

print("🔁 Retraining decision evaluated.")
print("Retrain required:", retrain_required)
print("📝 Decision logged to:", RETRAIN_LOG_FILE)
