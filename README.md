# MLOps Portfolio: Stock Return Prediction with Drift Detection & Retraining

A production-like MLOps system demonstrating the full lifecycle: **train → serve → monitor → detect drift → automatically retrain → validate → promote**.

## Overview

This project trains a stock price return predictor on historical data (2018–2021), deploys it via FastAPI, monitors predictions with Prometheus/Grafana, detects when model performance degrades due to market regime shifts, and automatically retrains when data drift exceeds a threshold.

**Why finance?** Market regimes (COVID crash, 2022 bear market, 2023–2024 bull run) produce *real, natural data drift*, so the retraining story isn't simulated—it's genuine.

## Architecture

```
┌──────────────┐
│ Training Data│  (yfinance: SPY, AAPL, MSFT, 2018–2021)
│  2018–2021   │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Feature Eng +   │──→ MLflow Registry (Production)
│ XGBoost Train   │
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
       │ Evidently Drift      │
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
docker-compose up -d

# Wait ~30s for services to be ready
docker-compose logs -f api
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
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train.py
```

This:
- Downloads OHLCV data (2018–2024) via yfinance
- Trains on 2018–2021 only (to force drift when 2022–2024 is replayed)
- Logs metrics/model to MLflow
- Registers model as "Production" in the Model Registry

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
    "ticker": "SPY",
    "timestamp": "2024-01-15"
  }'
```

Response:
```json
{
  "prediction": 0.0012,
  "ticker": "SPY",
  "timestamp": "2024-01-15"
}
```

### 4. Simulate Production Traffic (Watch Drift Happen)

```bash
# Replay 2022–2024 data through the API (simulates 2+ years of real-time predictions)
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

### 6. Run Retraining Pipeline (Manual)

```bash
# Trigger the Prefect flow manually
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/pipeline/retrain_flow.py
```

This:
1. Loads latest data
2. Checks drift (optional)
3. Trains a new model on an updated window (e.g., 2018–2022)
4. Validates against 2022–2023 holdout
5. If metrics improve, registers and promotes to "Production"
6. FastAPI reloads the new model on next request

## Project Structure

```
mlops-finance/
├── src/
│   ├── data/
│   │   ├── fetch.py           # yfinance → parquet
│   │   └── features.py        # technical indicators + target
│   ├── train.py               # RandomForest + MLflow tracking/registry
│   ├── serve/
│   │   ├── app.py             # FastAPI /predict, /metrics
│   │   └── traffic_log.py     # log predictions (unused yet; future: detailed logging)
│   ├── monitoring/
│   │   └── drift.py           # Evidently drift detection
│   └── pipeline/
│       └── retrain_flow.py    # Prefect flow (retrain + validate + promote)
├── scripts/
│   └── simulate_traffic.py    # Replay historical data through API
├── prometheus/
│   └── prometheus.yml         # scrape /metrics from API
├── grafana/
│   └── dashboards/            # (dashboards added manually in UI)
├── tests/
│   ├── test_features.py       # feature engineering tests
│   └── test_train.py          # (placeholder for training tests)
├── Dockerfile
├── docker-compose.yml
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

### Why 2018–2021 training window?
- Deliberately excludes 2022 bear market + 2023–2024 bull run
- Ensures real drift when recent data is replayed
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

## References

- **MLflow**: https://mlflow.org
- **Prefect**: https://www.prefect.io
- **Evidently**: https://www.evidentlyai.com
- **FastAPI**: https://fastapi.tiangolo.com
- **Prometheus**: https://prometheus.io
- **Grafana**: https://grafana.com

---

**Author**: James Hourihane  
**Last Updated**: June 2026
