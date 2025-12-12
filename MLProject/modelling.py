import argparse
import os
from pathlib import Path
from contextlib import nullcontext

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_split(train_path: Path, test_path: Path, target_col: str):
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing in train data")
    if target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' missing in test data")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test


def train_and_log(X_train, X_test, y_train, y_test, experiment_name: str, random_state: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)
    # Use autolog for parameters/metrics but log models explicitly to ensure
    # the model artifact path is exactly `artifacts/model` so CI can find it.
    mlflow.sklearn.autolog(log_models=False)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    # Prevent MLflow Projects env run-id mismatch by clearing the env var, then only start a run if none is active.
    os.environ.pop("MLFLOW_RUN_ID", None)
    active = mlflow.active_run()
    run_ctx = nullcontext() if active else mlflow.start_run(run_name="ci_random_forest_autolog")

    with run_ctx:
        clf.fit(X_train, y_train)
        # Explicitly log the trained model to artifact path "model"
        # so `mlflow models build-docker -m runs:/<run_id>/model` can locate it.
        try:
            mlflow.sklearn.log_model(clf, artifact_path="model")
        except Exception as e:
            print(f"Warning: failed to mlflow.log_model: {e}")
        preds = clf.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        metrics_path = output_dir / "metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path)

        model_path = output_dir / "model.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train model via MLflow Project entrypoint")
    parser.add_argument("--train-path", type=Path, default=Path("data_clustering_preprocessing/train_processed.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data_clustering_preprocessing/test_processed.csv"))
    parser.add_argument("--target-col", type=str, default="Target")
    parser.add_argument("--experiment-name", type=str, default="ci_data_clustering")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    X_train, X_test, y_train, y_test = load_split(args.train_path, args.test_path, args.target_col)
    print(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    class_counts = np.bincount(y_train.to_numpy()) if y_train.dtype != object else y_train.value_counts().to_dict()
    print(f"Target distribution (train): {class_counts}")

    train_and_log(X_train, X_test, y_train, y_test, args.experiment_name, args.random_state, args.output_dir)


if __name__ == "__main__":
    main()
