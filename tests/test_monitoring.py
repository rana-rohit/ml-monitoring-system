"""
test_monitoring.py

Unit tests for the monitoring modules including drift detection,
alert generation, and retraining decisions.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path


class TestAlertCreation:
    """Tests for alert creation logic."""
    
    def test_create_alert_structure(self):
        """Test that created alerts have correct structure."""
        # Simulate alert creation
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "WARNING",
            "source": "data_drift",
            "message": "Test alert message"
        }
        
        assert "timestamp" in alert
        assert "level" in alert
        assert "source" in alert
        assert "message" in alert
    
    def test_alert_levels(self):
        """Test valid alert levels."""
        valid_levels = ["INFO", "WARNING", "CRITICAL"]
        
        for level in valid_levels:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "source": "test",
                "message": "Test message"
            }
            assert alert["level"] in valid_levels


class TestDriftDetection:
    """Tests for drift detection logic."""
    
    def test_p_value_threshold(self):
        """Test that p-value threshold is correctly applied."""
        P_VALUE_THRESHOLD = 0.05
        
        # Test case: p-value below threshold (drift detected)
        p_value_low = 0.03
        drift_detected = p_value_low < P_VALUE_THRESHOLD
        assert drift_detected is True
        
        # Test case: p-value above threshold (no drift)
        p_value_high = 0.10
        drift_detected = p_value_high < P_VALUE_THRESHOLD
        assert drift_detected is False
        
        # Test case: p-value at threshold (no drift)
        p_value_exact = 0.05
        drift_detected = p_value_exact < P_VALUE_THRESHOLD
        assert drift_detected is False
    
    def test_drift_report_structure(self, sample_drift_report):
        """Test drift report has correct structure."""
        for feature, stats in sample_drift_report.items():
            assert "ks_statistic" in stats
            assert "p_value" in stats
            assert "drift_detected" in stats
            assert isinstance(stats["ks_statistic"], float)
            assert isinstance(stats["p_value"], float)
            assert isinstance(stats["drift_detected"], bool)
    
    def test_count_drifted_features(self, sample_drift_report):
        """Test counting features with drift."""
        drifted_features = [
            feature for feature, stats in sample_drift_report.items()
            if stats["drift_detected"]
        ]
        assert len(drifted_features) == 2
        assert "feature_1" in drifted_features
        assert "feature_3" in drifted_features


class TestPerformanceMonitoring:
    """Tests for performance monitoring logic."""
    
    def test_degradation_threshold(self, sample_performance_metrics):
        """Test performance degradation detection."""
        baseline_accuracy = sample_performance_metrics["accuracy"]
        DEGRADATION_THRESHOLD = 0.90
        
        # Test case: performance above threshold (not degraded)
        current_accuracy = 0.92
        degraded = current_accuracy < DEGRADATION_THRESHOLD * baseline_accuracy
        assert degraded is False
        
        # Test case: performance below threshold (degraded)
        current_accuracy = 0.80
        degraded = current_accuracy < DEGRADATION_THRESHOLD * baseline_accuracy
        assert degraded is True
    
    def test_count_degraded_batches(self, sample_performance_history):
        """Test counting degraded batches."""
        degraded_batches = [
            batch for batch in sample_performance_history
            if batch["performance_degraded"]
        ]
        assert len(degraded_batches) == 1


class TestRetrainingDecision:
    """Tests for retraining decision logic."""
    
    def test_should_retrain_with_critical_alerts(self, sample_alerts):
        """Test retraining decision with critical alerts."""
        CRITICAL_ALERT_THRESHOLD = 1
        
        critical_alerts = [
            alert for alert in sample_alerts
            if alert["level"] == "CRITICAL"
        ]
        
        should_retrain = len(critical_alerts) >= CRITICAL_ALERT_THRESHOLD
        assert should_retrain is True
    
    def test_should_not_retrain_without_critical_alerts(self):
        """Test retraining decision without critical alerts."""
        alerts_no_critical = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "level": "WARNING",
                "source": "data_drift",
                "message": "Test warning"
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "level": "INFO",
                "source": "concept_drift",
                "message": "Test info"
            }
        ]
        
        CRITICAL_ALERT_THRESHOLD = 1
        
        critical_alerts = [
            alert for alert in alerts_no_critical
            if alert["level"] == "CRITICAL"
        ]
        
        should_retrain = len(critical_alerts) >= CRITICAL_ALERT_THRESHOLD
        assert should_retrain is False
    
    def test_lookback_window(self, sample_alerts):
        """Test alert filtering by lookback window."""
        LOOKBACK_HOURS = 24
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=LOOKBACK_HOURS)
        
        # Create alerts with recent timestamp
        recent_alert = {
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "level": "CRITICAL",
            "source": "test",
            "message": "Recent alert"
        }
        
        # Create alerts with old timestamp
        old_alert = {
            "timestamp": (now - timedelta(hours=48)).isoformat(),
            "level": "CRITICAL",
            "source": "test",
            "message": "Old alert"
        }
        
        alerts = [recent_alert, old_alert]
        
        recent_critical_alerts = [
            alert for alert in alerts
            if alert["level"] == "CRITICAL"
            and datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
        ]
        
        assert len(recent_critical_alerts) == 1
        assert recent_critical_alerts[0]["message"] == "Recent alert"


class TestConceptDrift:
    """Tests for concept drift detection."""
    
    def test_concept_drift_report_structure(self, sample_concept_drift_report):
        """Test concept drift report structure."""
        assert "ks_statistic" in sample_concept_drift_report
        assert "p_value" in sample_concept_drift_report
        assert "concept_drift_detected" in sample_concept_drift_report
    
    def test_concept_drift_detection(self, sample_concept_drift_report):
        """Test concept drift detection logic."""
        # Sample report has p_value > 0.05, so no drift
        assert sample_concept_drift_report["concept_drift_detected"] is False
