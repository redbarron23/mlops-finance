"""Prefect flow for automated retraining."""
import os
from datetime import datetime, timedelta

import mlflow
import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.metrics import r2_score

from data.fetch import fetch_stock_data, load_stock_data
from data.features import compute_features, get_feature_columns, get_target_column
from monitoring.drift import compute_drift_from_logs
from train import train_model, register_model_mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "stock_return_predictor"
DRIFT_THRESHOLD = 0.3


@task(name="fetch_latest_data")
def fetch_latest_data_task(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch latest market data."""
    logger = get_run_logger()
    logger.info(f"Fetching latest data for {tickers}")
    return fetch_stock_data(tickers)


@task(name="check_drift")
def check_drift_task() -> float:
    """Compute drift score from recent predictions."""
    logger = get_run_logger()
    drift_score = compute_drift_from_logs()
    logger.info(f"Drift score: {drift_score:.3f} (threshold: {DRIFT_THRESHOLD})")
    return drift_score


@task(name="prepare_data")
def prepare_data_task(
    tickers: list[str],
    train_end_date: str,
    test_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and prepare training + validation data."""
    logger = get_run_logger()

    all_train, all_test = [], []

    for ticker in tickers:
        df = load_stock_data(ticker)
        df = compute_features(df)

        train_df = df[df["date"] <= train_end_date]
        test_df = df[(df["date"] > train_end_date) & (df["date"] <= test_end_date)]

        all_train.append(train_df)
        all_test.append(test_df)

    combined_train = pd.concat(all_train, ignore_index=True)
    combined_test = pd.concat(all_test, ignore_index=True)

    feature_cols = get_feature_columns()
    target_col = get_target_column()

    X_train = combined_train[feature_cols]
    y_train = combined_train[target_col]
    X_test = combined_test[feature_cols]
    y_test = combined_test[target_col]

    logger.info(
        f"Train: {len(X_train)} samples, Test: {len(X_test)} samples"
    )

    return X_train, y_train, X_test, y_test, feature_cols


@task(name="train_candidate")
def train_candidate_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Train a new candidate model."""
    logger = get_run_logger()
    logger.info("Training candidate model...")
    result = train_model(X_train, y_train)
    logger.info(f"Candidate metrics: {result['metrics']}")
    return result


@task(name="evaluate_vs_production")
def evaluate_production_task(
    candidate: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Evaluate candidate vs current production model."""
    logger = get_run_logger()

    # Get current production model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)

    try:
        model_version = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )[0]
        prod_model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}/Production"
        )
        prod_r2 = r2_score(y_test, prod_model.predict(X_test))
    except Exception as e:
        logger.warning(f"Could not load production model: {e}")
        prod_r2 = -999

    candidate_model = candidate["model"]
    candidate_r2 = r2_score(y_test, candidate_model.predict(X_test))

    logger.info(
        f"Production R²: {prod_r2:.3f}, Candidate R²: {candidate_r2:.3f}"
    )

    return candidate_r2, prod_r2


@task(name="register_if_improved")
def register_candidate_task(
    candidate: dict,
    candidate_r2: float,
    prod_r2: float,
    feature_cols: list[str],
) -> bool:
    """Register candidate if it outperforms production."""
    logger = get_run_logger()

    if candidate_r2 > prod_r2:
        logger.info(
            f"Candidate improved! ({candidate_r2:.3f} > {prod_r2:.3f})"
        )
        register_model_mlflow(
            candidate["model"],
            candidate["metrics"],
            feature_cols,
            tags={
                "stage": "candidate",
                "date": datetime.now().isoformat(),
                "promotion_reason": "improved_metrics",
            },
        )
        return True
    else:
        logger.info("Candidate did not improve; not registering.")
        return False


@flow(name="stock_retrain_flow")
def retrain_flow(
    tickers: list[str] = ["GOOG", "NVDA", "ORCL", "MSFT"],
    train_end_date: str = "2022-12-31",
    test_end_date: str = "2023-12-31",
    drift_check: bool = True,
):
    """
    End-to-end retraining pipeline.

    Steps:
      1. Check drift; proceed only if threshold exceeded (or skip check)
      2. Prepare training data
      3. Train candidate model
      4. Evaluate vs production
      5. Register if improved
    """
    logger = get_run_logger()
    logger.info("=== Starting retrain flow ===")

    # Optionally check drift
    if drift_check:
        drift = check_drift_task()
        if drift < DRIFT_THRESHOLD:
            logger.info(
                f"Drift below threshold ({drift:.3f} < {DRIFT_THRESHOLD}); "
                "skipping retrain."
            )
            return

    # Fetch and prepare
    fetch_latest_data_task(tickers)
    X_train, y_train, X_test, y_test, feature_cols = prepare_data_task(
        tickers, train_end_date, test_end_date
    )

    # Train candidate
    candidate = train_candidate_task(X_train, y_train)

    # Evaluate
    candidate_r2, prod_r2 = evaluate_production_task(candidate, X_test, y_test)

    # Register if improved
    promoted = register_candidate_task(
        candidate, candidate_r2, prod_r2, feature_cols
    )

    logger.info(f"=== Retrain flow complete (promoted: {promoted}) ===")


if __name__ == "__main__":
    retrain_flow(
        train_end_date="2022-12-31",
        test_end_date="2023-12-31",
        drift_check=False,
    )
