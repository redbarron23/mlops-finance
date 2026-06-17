"""Train and register model with MLflow."""
import os
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.fetch import fetch_stock_data, load_stock_data
from src.data.features import compute_features, get_feature_columns, get_target_column
from src.data.demo_data import generate_all_synthetic_data

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MODEL_NAME = "stock_return_predictor"


def prepare_training_data(
    tickers: list[str],
    train_end_date: str = "2021-12-31",
    data_dir: str = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load stock data, compute features, filter to training period.

    Args:
      tickers: list of ticker symbols
      train_end_date: cutoff date for training data (inclusive)
      data_dir: path to raw data directory

    Returns:
      X_train, y_train (DataFrames), feature_columns (list)
    """
    all_dfs = []

    for ticker in tickers:
        df = load_stock_data(ticker, data_dir)
        df = compute_features(df)
        df = df[df["date"] <= train_end_date]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    feature_cols = get_feature_columns()
    target_col = get_target_column()

    X = combined[feature_cols]
    y = combined[target_col]

    print(f"Training data: {len(X)} samples across {len(tickers)} tickers")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")

    return X, y, feature_cols


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a RandomForest regressor and compute metrics.

    Returns dict with model, metrics, and feature importance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_importance": dict(
            zip(X.columns, model.feature_importances_)
        ),
    }


def register_model_mlflow(
    model,
    metrics: dict,
    feature_cols: list[str],
    tags: dict | None = None,
) -> str:
    """
    Log model and metrics to MLflow, register in Model Registry with 'production' stage.

    Returns model version URI.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("stock_return_prediction")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
            }
        )
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=np.array([1.0] * len(feature_cols)).reshape(1, -1),
        )

        run_id = run.info.run_id

    # Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered model: {MODEL_NAME} version {result.version}")

    # Transition to production
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
    )
    print(f"Transitioned {MODEL_NAME} v{result.version} to Production")

    return model_uri


def main():
    """Fetch data, train, and register baseline model."""
    # Ensure raw data exists
    data_dir = "data/raw"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("Generating synthetic stock data (yfinance unavailable)...")
        generate_all_synthetic_data()

    print("\n=== Training baseline model (2018-2021) ===")
    X, y, feature_cols = prepare_training_data(
        tickers=["GOOG", "NVDA", "ORCL", "MSFT"],
        train_end_date="2021-12-31",
    )

    print("\nTraining model...")
    result = train_model(X, y)
    model = result["model"]
    metrics = result["metrics"]

    print(f"\nMetrics:\n{metrics}")

    print("\nRegistering with MLflow...")
    register_model_mlflow(
        model,
        metrics,
        feature_cols,
        tags={"stage": "baseline", "date": datetime.now().isoformat()},
    )

    print("\n✓ Baseline model trained and registered")


if __name__ == "__main__":
    main()
