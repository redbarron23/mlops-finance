"""FastAPI serving app with Prometheus metrics."""
import os
from datetime import datetime
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "stock_return_predictor"
MODEL_STAGE = "Production"

app = FastAPI(title="Stock Return Predictor")

# Global model cache
_model = None
_feature_columns = None

# Prometheus metrics
predictions_total = Counter(
    "predictions_total", "Total predictions made", ["status"]
)
prediction_latency = Histogram(
    "prediction_latency_seconds", "Latency of predictions"
)
prediction_value = Gauge(
    "prediction_value", "Recent prediction value (latest)", ["ticker"]
)


class PredictionInput(BaseModel):
    """Request schema for /predict endpoint."""

    sma_20: float
    sma_50: float
    rsi_14: float
    volatility_20: float
    return_lag_1: float
    return_lag_2: float
    return_lag_3: float
    return_lag_4: float
    return_lag_5: float
    ticker: str = "SPY"
    timestamp: str | None = None


class PredictionOutput(BaseModel):
    """Response schema for /predict endpoint."""

    prediction: float
    ticker: str
    timestamp: str


def load_model():
    """Load model from MLflow (local file store)."""
    global _model, _feature_columns

    if _model is not None:
        return _model

    try:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Try to load from local file store (mlruns directory)
        import os as os_module
        mlruns_dir = "/app/mlruns"  # Inside container
        if not os_module.path.exists(mlruns_dir):
            mlruns_dir = "mlruns"  # Local development

        # Find the latest production model
        model_uri = f"file://{os_module.path.abspath(mlruns_dir)}"
        client = mlflow.MlflowClient(model_uri)

        # Try to load from local models directory
        import glob
        model_paths = glob.glob(f"{mlruns_dir}/**/MLmodel", recursive=True)
        if not model_paths:
            raise FileNotFoundError(f"No MLflow models found in {mlruns_dir}")

        # Use the first/latest model found
        model_path = os_module.path.dirname(model_paths[0])
        _model = mlflow.pyfunc.load_model(model_path)

        _feature_columns = [
            "sma_20",
            "sma_50",
            "rsi_14",
            "volatility_20",
            "return_lag_1",
            "return_lag_2",
            "return_lag_3",
            "return_lag_4",
            "return_lag_5",
        ]
        print(f"Loaded model from {model_path}")
        return _model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def log_prediction(input_data: dict, prediction: float):
    """Log prediction to local store for drift analysis."""
    log_dir = Path("data/predictions")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"predictions.parquet"
    record = {
        "timestamp": input_data.get("timestamp", datetime.now().isoformat()),
        "ticker": input_data["ticker"],
        "prediction": prediction,
        **{k: v for k, v in input_data.items() if k != "ticker"},
    }

    try:
        if log_file.exists():
            df = pd.read_parquet(log_file)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])

        df.to_parquet(log_file, index=False)
    except Exception as e:
        print(f"Warning: failed to log prediction: {e}")


@app.on_event("startup")
async def startup():
    """Log startup (model loads on first request)."""
    print("✓ API started (model will load on first request)")


@app.post("/predict")
async def predict(req: PredictionInput) -> PredictionOutput:
    """
    Make a prediction given input features.

    Features must match the training schema.
    """
    try:
        with prediction_latency.time():
            if _model is None:
                load_model()

            features = np.array(
                [
                    [
                        req.sma_20,
                        req.sma_50,
                        req.rsi_14,
                        req.volatility_20,
                        req.return_lag_1,
                        req.return_lag_2,
                        req.return_lag_3,
                        req.return_lag_4,
                        req.return_lag_5,
                    ]
                ]
            )

            pred = _model.predict(features)[0]

            timestamp = req.timestamp or datetime.now().isoformat()
            log_prediction(req.dict(), pred)
            prediction_value.labels(ticker=req.ticker).set(pred)
            predictions_total.labels(status="success").inc()

            return PredictionOutput(
                prediction=float(pred),
                ticker=req.ticker,
                timestamp=timestamp,
            )

    except Exception as e:
        predictions_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
