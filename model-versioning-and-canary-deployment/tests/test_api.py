"""
Comprehensive API tests for MLflow model serving with canary deployment
"""
import pytest
import requests
from collections import Counter

BASE_URL = "http://localhost:8000"
TIMEOUT = 5


@pytest.fixture
def skip_if_server_down():
    """Skip test if server is not running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code != 200:
            pytest.skip("API server not responding correctly")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pytest.skip("API server not running")


class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self, skip_if_server_down):
        """Test root endpoint returns status and canary info"""
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "canary_probability" in data
        assert "current_model_loaded" in data
        assert "next_model_loaded" in data
    
    def test_get_predict_endpoint(self, skip_if_server_down):
        """Test GET /predict returns constant value"""
        response = requests.get(f"{BASE_URL}/predict", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "y_pred" in data
        assert data["y_pred"] == 2


class TestPredictionEndpoint:
    """Test prediction functionality"""
    
    def test_post_predict_with_iris_data(self, skip_if_server_down):
        """Test POST /predict with Iris dataset features"""
        payload = {
            "data": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 3.4, 5.4, 2.3]
            ]
        }
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) == 2
            assert "model_used" in data
    
    def test_post_predict_single_sample(self, skip_if_server_down):
        """Test POST /predict with single sample"""
        payload = {"data": [[5.1, 3.5, 1.4, 0.2]]}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 1


class TestModelManagement:
    """Test model management endpoints"""
    
    def test_update_model_endpoint(self, skip_if_server_down):
        """Test /update-model endpoint updates next model"""
        payload = {"model_uri": "models:/tracking-quickstart/2"}
        response = requests.post(
            f"{BASE_URL}/update-model",
            json=payload,
            timeout=TIMEOUT
        )
        # Accept 200 (success) or 400 (model not found in MLflow)
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "Next model updated" in data["status"]
            assert "canary_probability" in data
    
    def test_accept_next_model_endpoint(self, skip_if_server_down):
        """Test /accept-next-model endpoint promotes next to current"""
        # First update next model
        payload = {"model_uri": "models:/tracking-quickstart/2"}
        requests.post(f"{BASE_URL}/update-model", json=payload, timeout=TIMEOUT)
        
        # Then accept it
        response = requests.post(
            f"{BASE_URL}/accept-next-model",
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "current_model_loaded" in data
            assert "next_model_loaded" in data


class TestCanaryDeployment:
    """Test canary deployment functionality"""
    
    def test_set_canary_probability(self, skip_if_server_down):
        """Test setting canary probability"""
        # Test various probability values
        for prob in [0.0, 0.5, 0.9, 1.0]:
            payload = {"probability": prob}
            response = requests.post(
                f"{BASE_URL}/set-canary-probability",
                json=payload,
                timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "canary_probability" in data
            assert data["canary_probability"] == prob
            assert "current_model_percentage" in data
            assert "next_model_percentage" in data
    
    def test_invalid_canary_probability(self, skip_if_server_down):
        """Test that invalid probabilities are rejected"""
        for invalid_prob in [-0.1, 1.5, 2.0]:
            payload = {"probability": invalid_prob}
            response = requests.post(
                f"{BASE_URL}/set-canary-probability",
                json=payload,
                timeout=TIMEOUT
            )
            assert response.status_code == 400
    
    def test_canary_traffic_distribution_100_current(self, skip_if_server_down):
        """Test that 100% probability routes all traffic to current model"""
        # Set to 100% current
        requests.post(
            f"{BASE_URL}/set-canary-probability",
            json={"probability": 1.0},
            timeout=TIMEOUT
        )
        
        # Make multiple predictions
        payload = {"data": [[5.1, 3.5, 1.4, 0.2]]}
        model_usage = Counter()
        
        for _ in range(20):
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_used = response.json().get("model_used", "unknown")
                model_usage[model_used] += 1
        
        # All requests should use current model (or fallback variations)
        assert model_usage.get("current", 0) + model_usage.get("current (fallback)", 0) >= 19
    
    def test_canary_traffic_distribution_split(self, skip_if_server_down):
        """Test that 50/50 split distributes traffic"""
        # Set to 50/50
        requests.post(
            f"{BASE_URL}/set-canary-probability",
            json={"probability": 0.5},
            timeout=TIMEOUT
        )
        
        # Make multiple predictions
        payload = {"data": [[5.1, 3.5, 1.4, 0.2]]}
        model_usage = Counter()
        
        for _ in range(50):
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_used = response.json().get("model_used", "unknown")
                model_usage[model_used] += 1
        
        # With 50 requests, we expect roughly 25 each (allow some variance)
        # At least both models should be used
        current_count = model_usage.get("current", 0) + model_usage.get("current (fallback)", 0)
        next_count = model_usage.get("next", 0) + model_usage.get("next (fallback)", 0)
        
        assert current_count > 0, "Current model should be used"
        assert next_count > 0, "Next model should be used"
        assert 15 <= current_count <= 35, f"Expected ~25 current model uses, got {current_count}"
        assert 15 <= next_count <= 35, f"Expected ~25 next model uses, got {next_count}"
    
    def test_canary_workflow(self, skip_if_server_down):
        """Test complete canary deployment workflow"""
        # Step 1: Update next model
        response = requests.post(
            f"{BASE_URL}/update-model",
            json={"model_uri": "models:/tracking-quickstart/2"},
            timeout=TIMEOUT
        )
        if response.status_code != 200:
            pytest.skip("Model not available in MLflow")
        
        # Step 2: Set to 10% canary
        response = requests.post(
            f"{BASE_URL}/set-canary-probability",
            json={"probability": 0.9},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        
        # Step 3: Make some predictions
        payload = {"data": [[5.1, 3.5, 1.4, 0.2]]}
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        assert response.status_code in [200, 503]
        
        # Step 4: Increase to 50%
        response = requests.post(
            f"{BASE_URL}/set-canary-probability",
            json={"probability": 0.5},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        
        # Step 5: Accept next model
        response = requests.post(f"{BASE_URL}/accept-next-model", timeout=TIMEOUT)
        assert response.status_code == 200
        
        # Step 6: Reset to 100% current
        response = requests.post(
            f"{BASE_URL}/set-canary-probability",
            json={"probability": 1.0},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


@pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_canary_probability_values(skip_if_server_down, probability):
    """Test various canary probability values"""
    response = requests.post(
        f"{BASE_URL}/set-canary-probability",
        json={"probability": probability},
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    data = response.json()
    assert data["canary_probability"] == probability


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

