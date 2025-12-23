"""
conftest.py

Pytest configuration and shared fixtures for the ML Monitoring System tests.
"""

import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def temp_reports_dir(tmp_path):
    """Create a temporary reports directory structure."""
    dirs = [
        tmp_path / "reports" / "alerts",
        tmp_path / "reports" / "baseline",
        tmp_path / "reports" / "monitoring",
        tmp_path / "reports" / "drift",
        tmp_path / "reports" / "retraining",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return tmp_path / "reports"


@pytest.fixture
def sample_alerts():
    """Sample alerts for testing."""
    return [
        {
            "timestamp": "2024-01-01T10:00:00",
            "level": "WARNING",
            "source": "data_drift",
            "message": "Data drift detected in 3 features."
        },
        {
            "timestamp": "2024-01-01T11:00:00",
            "level": "CRITICAL",
            "source": "performance_monitor",
            "message": "Performance degraded in 2 batches."
        },
        {
            "timestamp": "2024-01-01T12:00:00",
            "level": "INFO",
            "source": "concept_drift",
            "message": "No concept drift detected."
        }
    ]


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "roc_auc": 0.98
    }


@pytest.fixture
def sample_drift_report():
    """Sample drift report for testing."""
    return {
        "feature_1": {
            "ks_statistic": 0.15,
            "p_value": 0.03,
            "drift_detected": True
        },
        "feature_2": {
            "ks_statistic": 0.08,
            "p_value": 0.45,
            "drift_detected": False
        },
        "feature_3": {
            "ks_statistic": 0.22,
            "p_value": 0.01,
            "drift_detected": True
        }
    }


@pytest.fixture
def sample_concept_drift_report():
    """Sample concept drift report for testing."""
    return {
        "ks_statistic": 0.12,
        "p_value": 0.08,
        "concept_drift_detected": False
    }


@pytest.fixture
def sample_performance_history():
    """Sample performance history for testing."""
    return [
        {
            "timestamp": "2024-01-01T10:00:00",
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "roc_auc": 0.98,
            "performance_degraded": False
        },
        {
            "timestamp": "2024-01-01T11:00:00",
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.86,
            "roc_auc": 0.92,
            "performance_degraded": False
        },
        {
            "timestamp": "2024-01-01T12:00:00",
            "accuracy": 0.75,
            "precision": 0.74,
            "recall": 0.73,
            "roc_auc": 0.82,
            "performance_degraded": True
        }
    ]
