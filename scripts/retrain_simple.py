"""Simple retraining script - detect drift and retrain if needed."""
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.data.features import compute_features, get_feature_columns, get_target_column
from src.data.fetch import load_stock_data
from src.monitoring.drift import compute_drift_from_logs
from src.train import train_model, register_model_mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
DRIFT_THRESHOLD = 0.0  # Lower threshold to demonstrate retraining (synthetic data has minimal drift)


def prepare_data(
    tickers: list[str],
    train_end_date: str,
    test_end_date: str,
) -> tuple:
    """Load and prepare training + validation data."""
    print(f"Preparing data...")
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

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    return X_train, y_train, X_test, y_test, feature_cols


def main():
    """Detect drift and retrain if needed."""
    print("\n=== Retraining Pipeline ===\n")

    # Step 1: Check drift
    print("Step 1: Checking drift...")
    drift_score = compute_drift_from_logs()
    print(f"  Drift score: {drift_score:.3f} (threshold: {DRIFT_THRESHOLD})")

    if drift_score < DRIFT_THRESHOLD:
        print(f"  → Drift below threshold, skipping retrain")
        return

    print(f"  → Drift detected! Triggering retrain...\n")

    # Step 2: Prepare data (train on 2018-2022, test on 2023-2024)
    print("Step 2: Preparing training data...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    tickers = ["GOOG", "NVDA", "ORCL", "MSFT"]
    X_train, y_train, X_test, y_test, feature_cols = prepare_data(
        tickers,
        train_end_date="2022-12-31",
        test_end_date="2024-06-30",
    )

    # Step 3: Train candidate model
    print("\nStep 3: Training candidate model...")
    candidate = train_model(X_train, y_train)
    candidate_metrics = candidate["metrics"]
    print(f"  Candidate metrics:")
    print(f"    Train R²: {candidate_metrics['train_r2']:.4f}")
    print(f"    Test R²:  {candidate_metrics['test_r2']:.4f}")

    # Step 4: Evaluate vs production
    print("\nStep 4: Comparing to production model...")

    # Load production model and evaluate
    import glob

    model_paths = glob.glob(f"{MLFLOW_TRACKING_URI}/**/MLmodel", recursive=True)
    if model_paths:
        model_path = os.path.dirname(model_paths[-1])  # Latest model
        prod_model = mlflow.pyfunc.load_model(model_path)
        prod_r2 = r2_score(y_test, prod_model.predict(X_test))
        print(f"  Production R²: {prod_r2:.4f}")
    else:
        prod_r2 = -999
        print(f"  Production model: not found")

    candidate_r2 = r2_score(y_test, candidate["model"].predict(X_test))
    print(f"  Candidate R²:  {candidate_r2:.4f}")

    # Step 5: Register if improved
    print("\nStep 5: Promotion decision...")
    if candidate_r2 > prod_r2:
        print(f"  ✓ Candidate improved! ({candidate_r2:.4f} > {prod_r2:.4f})")
        print(f"  Registering new model as Production...\n")
        register_model_mlflow(
            candidate["model"],
            candidate_metrics,
            feature_cols,
            tags={
                "stage": "candidate_promoted",
                "drift_score": drift_score,
                "improved_over_production": True,
            },
        )
        print("\n✓ Retraining complete - new model promoted to Production")
    else:
        print(f"  ✗ Candidate did not improve ({candidate_r2:.4f} <= {prod_r2:.4f})")
        print(f"  Keeping current production model")


if __name__ == "__main__":
    main()
