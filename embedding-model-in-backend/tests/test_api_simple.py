"""
Simple API tests that don't require MLflow
"""
import pytest
import requests
import time


def test_api_running():
    """Test that API is accessible (requires running server)"""
    try:
        response = requests.get("http://localhost:8000/predict", timeout=2)
        assert response.status_code == 200
        assert response.json() == {"y_pred": 2}
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")
    except requests.exceptions.Timeout:
        pytest.skip("API server timeout")


def test_get_predict_endpoint():
    """Test GET /predict returns constant value"""
    try:
        response = requests.get("http://localhost:8000/predict", timeout=2)
        assert response.status_code == 200
        data = response.json()
        assert "y_pred" in data
        assert data["y_pred"] == 2
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")


def test_post_predict_endpoint():
    """Test POST /predict with valid data"""
    try:
        # Test data format: [size, bedrooms, garden]
        test_data = {
            "data": [[1500.0, 3, 1]]
        }
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=5
        )
        # Accept 200 (success) or 503 (model not available)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == 1
            # House price should be positive
            assert data["predictions"][0] > 0
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")


def test_root_endpoint():
    """Test root endpoint"""
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")
