"""
config.py

Centralized configuration for the ML Model Monitoring System.
All file paths and thresholds are defined here for easy maintenance.
"""

from pathlib import Path

# =========================
# BASE DIRECTORIES
# =========================

# Project root directory
BASE_DIR = Path(__file__).parent.parent

# Main directories
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

# =========================
# MODEL PATHS
# =========================

BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
BASELINE_MODEL_PATH = BASELINE_MODEL_DIR / "model.joblib"

# =========================
# REPORT PATHS
# =========================

# Baseline reports
BASELINE_REPORTS_DIR = REPORTS_DIR / "baseline"
BASELINE_METRICS_PATH = BASELINE_REPORTS_DIR / "performance_metrics.json"
BASELINE_FEATURE_STATS_PATH = BASELINE_REPORTS_DIR / "feature_stats.json"

# Monitoring reports
MONITORING_REPORTS_DIR = REPORTS_DIR / "monitoring"
LATEST_PERFORMANCE_PATH = MONITORING_REPORTS_DIR / "latest_performance.json"
PERFORMANCE_HISTORY_PATH = MONITORING_REPORTS_DIR / "performance_history.json"
CONCEPT_DRIFT_PATH = MONITORING_REPORTS_DIR / "concept_drift_report.json"

# Drift reports
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
DATA_DRIFT_PATH = DRIFT_REPORTS_DIR / "data_drift_report.json"

# Alert reports
ALERTS_DIR = REPORTS_DIR / "alerts"
ALERTS_LOG_PATH = ALERTS_DIR / "alerts_log.json"

# Retraining reports
RETRAINING_DIR = REPORTS_DIR / "retraining"
RETRAIN_DECISIONS_PATH = RETRAINING_DIR / "retrain_decisions.json"

# =========================
# THRESHOLDS & PARAMETERS
# =========================

# Data drift detection
P_VALUE_THRESHOLD = 0.05  # Statistical significance level for KS test

# Performance monitoring
BATCH_SIZE = 50  # Number of samples per monitoring batch
DEGRADATION_THRESHOLD = 0.90  # Performance degradation threshold (% of baseline)

# Retraining controller
CRITICAL_ALERT_THRESHOLD = 1  # Number of critical alerts before retraining
LOOKBACK_HOURS = 24  # Time window for alert analysis (hours)

# =========================
# API CONFIGURATION
# =========================

API_HOST = "0.0.0.0"
API_PORT = 8000

# =========================
# HELPER FUNCTIONS
# =========================

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        MODELS_DIR,
        BASELINE_MODEL_DIR,
        REPORTS_DIR,
        BASELINE_REPORTS_DIR,
        MONITORING_REPORTS_DIR,
        DRIFT_REPORTS_DIR,
        ALERTS_DIR,
        RETRAINING_DIR,
        DATA_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_path_string(path: Path) -> str:
    """Convert Path to string for compatibility with json.load/dump."""
    return str(path)
