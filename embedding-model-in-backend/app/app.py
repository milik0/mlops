from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional, Any
import os

# Model path from environment variable or default to /models/regression.joblib for Docker volume mount
MODEL_PATH = os.getenv("MODEL_PATH", "/models/regression.joblib")

# Fallback to local paths if the default doesn't exist (for local development)
if not os.path.exists(MODEL_PATH):
    if os.path.exists("regression.joblib"):
        MODEL_PATH = "regression.joblib"
    else:
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "regression.joblib")

# Load the model at startup
model: Optional[Any] = None

def load_model_from_file(path: str):
    """Attempt to load a model from joblib file and return it, or None on failure."""
    global model
    try:
        loaded = joblib.load(path)
        model = loaded
        print(f"[app] Model loaded successfully from {path}")
        return model
    except Exception as e:
        # Keep model as None and log the error for debugging (don't raise here)
        print(f"[app] Failed to load model from {path}: {e}")
        return None

# Try to load the model at startup
load_model_from_file(MODEL_PATH)

app = FastAPI()

class PredictRequest(BaseModel):
    data: list  # 2D list of feature values

@app.get("/")
def root():
    return {"status": "FastAPI service running"}


@app.get("/predict")
def predict_get():
    """Exercise 1: simple GET /predict that returns a constant prediction.

    This satisfies the initial exercise requirement which asks for a GET
    endpoint returning {"y_pred": 2} so it can be tested from browsers and
    simple HTTP clients.
    """
    return {"y_pred": 2}

@app.post("/predict")
def predict(request: PredictRequest):
    global model
    try:
        # If model is not loaded, try to load it once.
        if model is None:
            if load_model_from_file(MODEL_PATH) is None:
                # Service temporary unavailable because no model is loaded
                raise HTTPException(status_code=503, detail="Model not available")

        df = pd.DataFrame(request.data)
        preds = model.predict(df)

        # Ensure predictions are JSON serializable
        try:
            preds_list = preds.tolist()
        except Exception:
            preds_list = list(map(float, preds))

        return {"predictions": preds_list}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateModelRequest(BaseModel):
    model_path: str

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """Load a new model from the provided file path and replace the in-memory model.

    Returns a 400 response if loading fails.
    """
    global model
    try:
        loaded = load_model_from_file(request.model_path)
        if loaded is None:
            raise HTTPException(status_code=400, detail=f"Failed to load model from {request.model_path}")
        return {"status": f"Model updated from {request.model_path}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
