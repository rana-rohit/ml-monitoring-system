"""
test_api.py

Unit tests for the FastAPI backend endpoints.
"""

import json
import pytest
from pathlib import Path


class TestLoadJson:
    """Tests for the load_json helper function."""
    
    def test_load_json_existing_file(self, tmp_path):
        """Test loading an existing JSON file."""
        # Import here to avoid import issues
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.api.main import load_json
        
        # Create a test JSON file
        test_data = {"key": "value", "number": 42}
        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        # Test loading
        result = load_json(str(test_file))
        assert result == test_data
    
    def test_load_json_nonexistent_file(self, tmp_path):
        """Test loading a non-existent file returns None."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.api.main import load_json
        
        result = load_json(str(tmp_path / "nonexistent.json"))
        assert result is None


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check(self):
        """Test health check returns ok status."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.api.main import health_check
        
        result = health_check()
        assert result["status"] == "ok"
        assert result["service"] == "ml-model-monitoring-api"


class TestAPIPathsConfiguration:
    """Tests for API path configuration."""
    
    def test_paths_dict_contains_required_keys(self):
        """Test that PATHS dictionary contains all required keys."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.api.main import PATHS
        
        required_keys = [
            "alerts",
            "baseline_metrics",
            "latest_metrics",
            "performance_history",
            "data_drift",
            "concept_drift",
            "retrain_status"
        ]
        
        for key in required_keys:
            assert key in PATHS, f"Missing key: {key}"
