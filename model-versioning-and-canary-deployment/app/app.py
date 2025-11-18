from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc as pyfunc
from typing import Optional, Any, cast
import random

mlflow.set_tracking_uri("http://mlflow:8080")
MLFLOW_MODEL_URI = "models:/tracking-quickstart/latest"

# Canary deployment: two model slots
current_model: Optional[Any] = None
next_model: Optional[Any] = None
canary_probability: float = 1.0  # Probability to use current model (1.0 = 100% current)

def load_model_from_uri(uri: str):
    """Attempt to load a model from MLflow and return it, or None on failure."""
    try:
        loaded = pyfunc.load_model(uri)
        return loaded
    except Exception as e:
        print(f"[app] Failed to load model from {uri}: {e}")
        return None

# Try a best-effort load of the default model for both slots
loaded = load_model_from_uri(MLFLOW_MODEL_URI)
current_model = loaded
next_model = loaded

app = FastAPI()

class PredictRequest(BaseModel):
    data: list  # 2D list of feature values

@app.get("/")
def root():
    return {
        "status": "FastAPI MLflow service running",
        "canary_probability": canary_probability,
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None
    }


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
    global current_model, next_model
    try:
        # Select model based on canary probability
        use_current = random.random() < canary_probability
        selected_model = current_model if use_current else next_model
        model_type = "current" if use_current else "next"
        
        # If selected model is not loaded, try to load default or use the other model
        if selected_model is None:
            if use_current and next_model is not None:
                selected_model = next_model
                model_type = "next (fallback)"
            elif not use_current and current_model is not None:
                selected_model = current_model
                model_type = "current (fallback)"
            else:
                # Try to load default model
                loaded = load_model_from_uri(MLFLOW_MODEL_URI)
                if loaded is None:
                    raise HTTPException(status_code=503, detail="No model available")
                selected_model = loaded
                current_model = loaded
                next_model = loaded
                model_type = "default"

        df = pd.DataFrame(request.data)
        m = cast(Any, selected_model)
        preds = m.predict(df)

        # Ensure predictions are JSON serializable
        try:
            preds_list = preds.tolist()
        except Exception:
            preds_list = list(map(float, preds))

        return {
            "predictions": preds_list,
            "model_used": model_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateModelRequest(BaseModel):
    model_uri: str

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """Load a new model from the provided URI and set it as the next model.
    
    This allows canary testing of the new model before promoting it to current.
    """
    global next_model
    try:
        loaded = load_model_from_uri(request.model_uri)
        if loaded is None:
            raise HTTPException(status_code=400, detail=f"Failed to load model from {request.model_uri}")
        next_model = loaded
        return {
            "status": f"Next model updated from {request.model_uri}",
            "canary_probability": canary_probability
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/accept-next-model")
def accept_next_model():
    """Promote the next model to current model.
    
    Both current and next will point to the same model after this operation.
    """
    global current_model, next_model
    if next_model is None:
        raise HTTPException(status_code=400, detail="No next model available to accept")
    
    current_model = next_model
    return {
        "status": "Next model promoted to current. Both slots now use the same model.",
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None
    }


class SetCanaryProbabilityRequest(BaseModel):
    probability: float

@app.post("/set-canary-probability")
def set_canary_probability(request: SetCanaryProbabilityRequest):
    """Set the probability of using the current model vs next model.
    
    Args:
        probability: Value between 0 and 1. 
                    1.0 = 100% current model
                    0.5 = 50% current, 50% next
                    0.0 = 100% next model
    """
    global canary_probability
    if not 0 <= request.probability <= 1:
        raise HTTPException(status_code=400, detail="Probability must be between 0 and 1")
    
    canary_probability = request.probability
    return {
        "status": "Canary probability updated",
        "canary_probability": canary_probability,
        "current_model_percentage": f"{canary_probability * 100:.1f}%",
        "next_model_percentage": f"{(1 - canary_probability) * 100:.1f}%"
    }
