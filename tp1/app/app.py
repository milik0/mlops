from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc as pyfunc
from typing import Optional, Any, cast

mlflow.set_tracking_uri("http://mlflow:8080")
MLFLOW_MODEL_URI = "models:/tracking-quickstart/2"

# Do not load the model at import time to avoid crashing the app when MLflow
# is not available (e.g., running locally without MLflow server).
model: Optional[Any] = None

def load_model_from_uri(uri: str):
    """Attempt to load a model from MLflow and return it, or None on failure."""
    global model
    try:
        loaded = pyfunc.load_model(uri)
        model = loaded
        return model
    except Exception as e:
        # Keep model as None and log the error for debugging (don't raise here)
        print(f"[app] Failed to load model from {uri}: {e}")
        return None

# Try a best-effort load of the default model but don't fail the import if it fails.
load_model_from_uri(MLFLOW_MODEL_URI)

app = FastAPI()

class PredictRequest(BaseModel):
    data: list  # 2D list of feature values

@app.get("/")
def root():
    return {"status": "FastAPI MLflow service running"}


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
        # If model is not loaded, try to load the default model once.
        if model is None:
            if load_model_from_uri(MLFLOW_MODEL_URI) is None:
                # Service temporary unavailable because no model is loaded
                raise HTTPException(status_code=503, detail="Model not available")

        df = pd.DataFrame(request.data)
        # mypy/linters can't always infer the None-check above, so cast for safety
        m = cast(Any, model)
        preds = m.predict(df)

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
    model_uri: str

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """Load a new model from the provided URI and replace the in-memory model.

    Returns a 400 response if loading fails.
    """
    global model
    try:
        loaded = load_model_from_uri(request.model_uri)
        if loaded is None:
            raise HTTPException(status_code=400, detail=f"Failed to load model from {request.model_uri}")
        return {"status": f"Model updated from {request.model_uri}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
