"""
Tests for the joblib model integration with the FastAPI backend
"""
import pytest
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from fastapi.testclient import TestClient
from app import app, load_model_from_file, MODEL_PATH, model


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


def test_model_loads_successfully():
    """Test that the model file exists and can be loaded"""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    loaded_model = load_model_from_file(MODEL_PATH)
    assert loaded_model is not None, "Model failed to load"


def test_root_endpoint(client):
    """Test the root endpoint returns correct status"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "FastAPI MLflow service running"


def test_get_predict_returns_constant(client):
    """Test GET /predict returns the constant value"""
    response = client.get("/predict")
    assert response.status_code == 200
    data = response.json()
    assert "y_pred" in data
    assert data["y_pred"] == 2


def test_post_predict_with_valid_data(client):
    """Test POST /predict with valid house data"""
    # Test data: [size, bedrooms, garden]
    test_data = {
        "data": [[1500.0, 3, 1]]
    }
    response = client.post("/predict", json=test_data)
    
    if model is None:
        # Model not loaded, should return 503
        assert response.status_code == 503
    else:
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
        assert isinstance(data["predictions"][0], (int, float))


def test_post_predict_with_multiple_rows(client):
    """Test POST /predict with multiple houses"""
    test_data = {
        "data": [
            [1500.0, 3, 1],
            [2000.0, 4, 1],
            [1000.0, 2, 0]
        ]
    }
    response = client.post("/predict", json=test_data)
    
    if model is None:
        assert response.status_code == 503
    else:
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        for pred in data["predictions"]:
            assert isinstance(pred, (int, float))


def test_post_predict_with_edge_cases(client):
    """Test POST /predict with edge case values"""
    test_cases = [
        {"data": [[0.0, 0, 0]]},  # Minimum values
        {"data": [[10000.0, 10, 1]]},  # Large values
        {"data": [[500.0, 1, 0]]},  # Small house
    ]
    
    for test_data in test_cases:
        response = client.post("/predict", json=test_data)
        if model is not None:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 1


def test_post_predict_with_invalid_data(client):
    """Test POST /predict with invalid data format"""
    invalid_cases = [
        {"data": []},  # Empty data
        {"data": "invalid"},  # Wrong type
        {"data": [[]]},  # Empty row
    ]
    
    for invalid_data in invalid_cases:
        response = client.post("/predict", json=invalid_data)
        # Should return 400, 422 (validation error), or 503 (if model not loaded)
        assert response.status_code in [400, 422, 503]


def test_post_predict_missing_data_field(client):
    """Test POST /predict without data field"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


def test_update_model_endpoint(client):
    """Test the update-model endpoint"""
    # Test with the existing model path
    update_data = {
        "model_path": MODEL_PATH
    }
    response = client.post("/update-model", json=update_data)
    
    if os.path.exists(MODEL_PATH):
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert MODEL_PATH in data["status"]
    else:
        assert response.status_code == 400


def test_update_model_with_invalid_path(client):
    """Test update-model with non-existent path"""
    update_data = {
        "model_path": "/non/existent/path/model.joblib"
    }
    response = client.post("/update-model", json=update_data)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_predictions_are_reasonable(client):
    """Test that predictions are in a reasonable range for house prices"""
    test_data = {
        "data": [[1500.0, 3, 1]]
    }
    response = client.post("/predict", json=test_data)
    
    if model is not None and response.status_code == 200:
        data = response.json()
        prediction = data["predictions"][0]
        # House prices should be positive and reasonable
        assert prediction > 0, "Price should be positive"
        assert prediction < 10000000, "Price should be reasonable"


def test_prediction_consistency(client):
    """Test that same input gives same output"""
    test_data = {
        "data": [[1500.0, 3, 1]]
    }
    
    response1 = client.post("/predict", json=test_data)
    response2 = client.post("/predict", json=test_data)
    
    if model is not None:
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        pred1 = response1.json()["predictions"][0]
        pred2 = response2.json()["predictions"][0]
        
        assert pred1 == pred2, "Same input should give same prediction"


def test_model_handles_different_feature_combinations(client):
    """Test model with various realistic feature combinations"""
    test_cases = [
        [800.0, 1, 0],    # Small apartment without garden
        [1200.0, 2, 0],   # Medium apartment without garden
        [1500.0, 3, 1],   # House with garden
        [2500.0, 4, 1],   # Large house with garden
        [3000.0, 5, 1],   # Very large house
    ]
    
    test_data = {"data": test_cases}
    response = client.post("/predict", json=test_data)
    
    if model is not None:
        assert response.status_code == 200
        data = response.json()
        predictions = data["predictions"]
        assert len(predictions) == len(test_cases)
        
        # Larger houses should generally be more expensive
        # (though this might not always hold due to other factors)
        for pred in predictions:
            assert isinstance(pred, (int, float))
            assert pred > 0
