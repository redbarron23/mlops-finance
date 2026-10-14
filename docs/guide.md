# MLOps Portfolio: Stock Return Prediction with Drift Detection & Retraining

An MLOps prototype for exploring **train → serve → monitor → compare drift → retrain → evaluate candidates**. See the README for the implemented boundaries and limitations.

## Overview

The modules demonstrate stock-return feature engineering, RandomForest training, FastAPI serving, prediction metrics, feature-distribution tests, and candidate retraining. The default baseline uses generated data; the service and retraining walkthrough below is experimental. See [the current limits](../README.md#quality-and-limits) before relying on the end-to-end integration.

**Why finance?** Different market regimes provide a useful context for learning about drift. The default training command generates synthetic OHLCV data when no raw data exists; the yfinance fetch helper is a separate, explicit path.

> This is a demo of MLOps mechanics (serving, monitoring, drift detection, automated retrain and promote), not a profitable trading model. Don't expect trading-grade accuracy.

## Architecture

```
┌──────────────┐
│ Training Data│  (yfinance: GOOG, NVDA, ORCL, MSFT)
│  2018–2021   │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Feature Eng +   │──→ MLflow Registry (Production)
│ RandomForest    │
└─────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ FastAPI Serving + Logging        │
│  (loads model from MLflow)       │
│  POST /predict → log features    │
│  GET /metrics → Prometheus       │
└──────────────────────────────────┘
       │
       ├──→ Prometheus scrapes metrics
       │    (prediction volume, latency, value)
       │
       └──→ Logs predictions to parquet
            for drift analysis
            │
            ▼
       ┌──────────────────────┐
       │ KS-test Drift Score  │
       │ (feature distribution│
       │  vs reference)       │
       └──────────────────────┘
            │
            ▼
       ┌──────────────────────┐
       │ Prefect Flow:        │
       │ - Check drift        │
       │ - Retrain if needed  │
       │ - Validate           │
       │ - Promote if better  │
       └──────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- ~2GB disk space (for data + artifacts)

### 1. Build & Start Services

```bash
# Clone/enter project
cd mlops-finance

# Build and start all services
docker compose up -d

# Wait ~30s for services to be ready
docker compose logs -f api
```

Services:
- **MLflow** (tracking + registry): http://localhost:5000
- **API** (FastAPI): http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 2. Fetch Data & Train Baseline Model

```bash
# From your local machine (not in container)
python -m pip install -e .
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.train
```

This:
- Generates synthetic OHLCV data (2018–2024) when `data/raw/` is empty
- Trains on 2018–2021 only (to force drift when 2022–2024 is replayed)
- Logs metrics/model to MLflow
- Logs artifacts in the local MLflow store and requests registration in the Model Registry

### 3. Make Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sma_20": 420.5,
    "sma_50": 415.0,
    "rsi_14": 55.0,
    "volatility_20": 0.015,
    "return_lag_1": 0.002,
    "return_lag_2": -0.001,
    "return_lag_3": 0.003,
    "return_lag_4": -0.002,
    "return_lag_5": 0.001,
    "ticker": "MSFT",
    "timestamp": "2024-01-15"
  }'
```

Response:
```json
{
  "prediction": 0.0012,
  "ticker": "MSFT",
  "timestamp": "2024-01-15"
}
```

### 4. Simulate Production Traffic (Watch Drift Happen)

```bash
# Replay the available 2022–2024 data through the API
# Because the model was trained only on 2018–2021 data, drift will be visible
python scripts/simulate_traffic.py
```

This simulates ~600+ days of trading data flowing through `/predict`, logging predictions for drift analysis.

### 5. Monitor Drift in Grafana

1. Open http://localhost:3000 (admin/admin)
2. **Data Source**: Prometheus (http://prometheus:9090)
3. Create dashboard or use example panels:
   - **Prediction Volume**: `predictions_total`
   - **Prediction Latency**: `prediction_latency_seconds`
   - **Drift Score** (once computed): custom metric from monitoring

### 6. Candidate Retraining Flow (Experimental)

```bash
# Trigger the Prefect flow manually
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.pipeline.retrain_flow
```

The intended flow is:
1. Loads latest data
2. Checks drift (optional)
3. Trains a new model on an updated window (e.g., 2018–2022)
4. Validates against 2022–2023 holdout
5. If metrics improve, registers and promotes to "Production"
6. Restart the API to clear its cached model after an artifact change. The current file-store loader does not select artifacts by registry stage.

## Project Structure

```
mlops-finance/
├── src/
│   ├── data/
│   │   ├── fetch.py           # yfinance → parquet
│   │   └── features.py        # technical indicators + target
│   ├── train.py               # RandomForest + MLflow tracking/registry
│   ├── serve/
│   │   └── app.py             # FastAPI /predict, /metrics
│   ├── monitoring/
│   │   └── drift.py           # KS-test drift score (scipy)
│   └── pipeline/
│       └── retrain_flow.py    # Prefect flow (retrain + validate + promote)
├── scripts/
│   └── simulate_traffic.py    # Replay historical data through API
├── prometheus/
│   └── prometheus.yml         # scrape /metrics from API
├── tests/
│   └── test_features.py       # feature engineering tests
├── Dockerfile
├── docker compose.yml
├── pyproject.toml
└── README.md
```

## Key Decisions

### Why Prefect over Airflow?
- Lighter weight, easier to run locally
- Modern Python-native API (fewer YAML files)
- Sufficient for a portfolio project

### Why RandomForest?
- Interpretable, no hyperparameter tuning needed
- Fast to train, good baseline
- Feature importance for explainability

### Why a KS test instead of Evidently?
- Per-feature two-sample Kolmogorov–Smirnov tests (scipy) are simple, transparent and dependency-light
- The drift score is the share of features with p < 0.05; the retrain flow triggers at 0.3
- Evidently reports are a possible future addition (see below)

### Why 2018–2021 training window?
- Deliberately excludes 2022 bear market + 2023–2024 bull run
- Provides separate time windows for exploring changes in feature distributions
- Justifies retraining story

## Testing

```bash
pytest tests/
```

## Monitoring & Observability

- **MLflow UI**: http://localhost:5000 — view runs, models, artifacts
- **Prometheus**: http://localhost:9090 — query metrics
- **Grafana**: http://localhost:3000 — visualize drift, prediction volume, latency

## Future Enhancements

- Schedule the retrain flow (e.g., weekly via Prefect scheduler)
- Add A/B testing: route % of traffic to candidate model
- Production metrics dashboard in Grafana (pre-built JSON)
- Integration tests (mock MLflow, test full flow)
- Cloud deployment (AWS ECS, SageMaker)
- Model explainability (SHAP, permutation importance)
- Evidently drift reports alongside the KS-test score
- Broader integration tests for training, serving, and the retrain flow (feature tests now run in CI)

## References

- **MLflow**: https://mlflow.org
- **Prefect**: https://www.prefect.io
- **FastAPI**: https://fastapi.tiangolo.com
- **Prometheus**: https://prometheus.io
- **Grafana**: https://grafana.com

---

**Author**: James Hourihane  
**Last Updated**: October 2026
