# MLflow Model Versioning and Deployment

This project demonstrates model lifecycle management using MLflow with a FastAPI web service for model serving.

## Project Structure

```
.
├── app/
│   ├── app.py              # FastAPI web service
│   ├── Dockerfile          # Docker image for the API service
│   └── requirements.txt    # Python dependencies
├── mlflow/
│   ├── poc.py              # Model training script
│   ├── mlruns/             # MLflow tracking data (generated)
│   └── artifacts/          # Model artifacts (generated)
├── tests/
│   └── test_api.py         # Comprehensive pytest test suite
├── docker-compose.yml      # Orchestrates MLflow and API services
└── answers.md              # Project documentation
```

## Features

### Part 1: Model Training and Tracking
- Model training on Iris dataset
- Hyperparameter tracking
- Metric tracking (accuracy)
- Model versioning in MLflow
- Multiple training runs with different hyperparameters

### Part 2: Model Serving
- FastAPI web service
- `/predict` endpoint for inference
- `/update-model` endpoint for dynamic model updates
- Model loaded from MLflow (not copied in Docker image)
- Automated tests
- Docker containerization with docker-compose

### Part 3: Canary Deployment
- Two model slots: `current` and `next`
- Probability-based traffic routing
- `/update-model` updates the next model
- `/accept-next-model` promotes next to current
- `/set-canary-probability` controls traffic split
- Gradual rollout with configurable percentages

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)

## Quick Start

### 1. Start the Services

```bash
docker-compose up --build
```

This will start:
- **MLflow UI**: http://localhost:8080
- **FastAPI service**: http://localhost:8000

### 2. Train and Register Models

Execute the training script inside the MLflow container:

```bash
docker exec -it mlflow python /mlflow/poc.py
```

This will:
- Train a Logistic Regression model on the Iris dataset
- Log hyperparameters and metrics to MLflow
- Register the model with name `tracking-quickstart`

You can modify hyperparameters in `mlflow/poc.py` and rerun to create multiple versions.

### 3. View Models in MLflow UI

Open http://localhost:8080 to:
- View all training runs
- Compare hyperparameters and metrics
- Access registered models
- Browse model artifacts

### 4. Test the API

#### Run Pytest Test Suite

The comprehensive test suite includes:
- Basic endpoint tests
- Prediction functionality tests
- Model management tests
- Canary deployment tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test classes:
```bash
# Test only basic endpoints
pytest tests/test_api.py::TestBasicEndpoints -v

# Test only canary deployment
pytest tests/test_api.py::TestCanaryDeployment -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

### 5. Update the Model

To switch to a different model version:

```bash
curl -X POST http://localhost:8000/update-model \
  -H "Content-Type: application/json" \
  -d '{"model_uri": "models:/tracking-quickstart/1"}'
```

Replace `1` with the desired version number from MLflow.

## API Endpoints

### GET /
Health check endpoint

**Response:**
```json
{
  "status": "FastAPI MLflow service running",
  "canary_probability": 1.0,
  "current_model_loaded": true,
  "next_model_loaded": true
}
```

### GET /predict
Simple prediction endpoint returning a constant (for testing)

**Response:**
```json
{"y_pred": 2}
```

### POST /predict
Make predictions using the loaded model

**Request:**
```json
{
  "data": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

**Response:**
```json
{
  "predictions": [0, 2],
  "model_used": "current"
}
```

### POST /update-model
Update the **next** model to a different version (for canary deployment)

**Request:**
```json
{
  "model_uri": "models:/tracking-quickstart/2"
}
```

**Response:**
```json
{
  "status": "Next model updated from models:/tracking-quickstart/2",
  "canary_probability": 1.0
}
```

### POST /accept-next-model
Promote the next model to current (both slots will use the same model)

**Response:**
```json
{
  "status": "Next model promoted to current. Both slots now use the same model.",
  "current_model_loaded": true,
  "next_model_loaded": true
}
```

### POST /set-canary-probability
Set the traffic split between current and next models

**Request:**
```json
{
  "probability": 0.7
}
```
- `1.0` = 100% current model
- `0.7` = 70% current, 30% next
- `0.5` = 50% current, 50% next
- `0.0` = 100% next model

**Response:**
```json
{
  "status": "Canary probability updated",
  "canary_probability": 0.7,
  "current_model_percentage": "70.0%",
  "next_model_percentage": "30.0%"
}
```

## Development

### Canary Deployment Workflow

The service implements canary deployment with two model slots:

1. **Load new model to next slot:**
```bash
curl -X POST http://localhost:8000/update-model \
  -H "Content-Type: application/json" \
  -d '{"model_uri": "models:/tracking-quickstart/3"}'
```

2. **Gradually increase traffic to next model:**
```bash
# Start with 10% traffic to next model
curl -X POST http://localhost:8000/set-canary-probability \
  -H "Content-Type: application/json" \
  -d '{"probability": 0.9}'

# Increase to 50% if metrics look good
curl -X POST http://localhost:8000/set-canary-probability \
  -H "Content-Type: application/json" \
  -d '{"probability": 0.5}'

# Finally go to 100% next model
curl -X POST http://localhost:8000/set-canary-probability \
  -H "Content-Type: application/json" \
  -d '{"probability": 0.0}'
```

3. **Promote next model to current:**
```bash
curl -X POST http://localhost:8000/accept-next-model
```

4. **Reset to 100% current model:**
```bash
curl -X POST http://localhost:8000/set-canary-probability \
  -H "Content-Type: application/json" \
  -d '{"probability": 1.0}'
```

This allows safe, gradual rollout of new models while monitoring their performance.

### Local Setup (without Docker)

1. Install dependencies:
```bash
pip install -r app/requirements.txt
```

2. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 8080 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts
```

3. Run the API:
```bash
cd app
uvicorn app:app --reload
```

### Training Different Models

Edit `mlflow/poc.py` to change hyperparameters:

```python
params = {
    "solver": "lbfgs",
    "max_iter": 1000,  # Change this
    "multi_class": "auto",
    "random_state": 8888,
}
```

Then rerun the training script.

## Architecture Highlights

1. **No Model in Docker Image**: The Dockerfile does NOT use `COPY` to include trained models. Models are loaded dynamically from MLflow at runtime.

2. **Model Versioning**: All models are tracked and versioned in MLflow, allowing easy rollback and comparison.

3. **Canary Deployment**: Two model slots (current and next) enable safe, gradual rollout of new models with configurable traffic splitting.

4. **Dynamic Updates**: The `/update-model` endpoint allows switching models without restarting the service or rebuilding containers.

5. **Separation of Concerns**: Training (MLflow) and serving (FastAPI) are independent services that communicate via the MLflow tracking server.

## Troubleshooting

### Service Not Starting
- Ensure ports 8000 and 8080 are available
- Check logs: `docker-compose logs`

### Model Not Loading
- Verify the model exists in MLflow UI
- Check the model URI format: `models:/MODEL_NAME/VERSION`
- Ensure MLflow service is running and accessible

### Prediction Errors
- Ensure input data matches the model's expected schema
- Check that a model is loaded (not `None`)

## Clean Up

Stop and remove containers:
```bash
docker-compose down
```

Remove volumes (this will delete MLflow data):
```bash
docker-compose down -v
```
