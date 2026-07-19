"""
train.py
--------
Etapa de entrenamiento y comparación de modelos del pipeline DVC.

Entrena dos modelos de clasificación (Regresión Logística y Random Forest)
para predecir target_opened, registra parámetros, métricas y artefactos
en MLflow, y guarda el mejor modelo (según ROC-AUC en test) en models/.

Uso:
    poetry run python src/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    params = load_params()
    train_params = params["train"]
    mlflow_params = params["mlflow"]

    processed_dir = Path(train_params["processed_dir"])
    models_dir = Path(train_params["models_dir"])
    outputs_dir = Path(train_params["outputs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    X_train = train_df.drop(columns=["target_opened"])
    y_train = train_df["target_opened"]
    X_test = test_df.drop(columns=["target_opened"])
    y_test = test_df["target_opened"]

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    results = {}

    # --- Modelo 1: Regresión Logística ---
    lr_params = train_params["logistic_regression"]
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params(lr_params)
        mlflow.log_param("model_type", "LogisticRegression")

        lr_model = LogisticRegression(
            max_iter=lr_params["max_iter"], C=lr_params["C"], random_state=train_params["seed"]
        )
        lr_model.fit(X_train, y_train)
        metrics = evaluate(lr_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr_model, "model")

        results["logistic_regression"] = metrics
        print("Logistic Regression:", metrics)

    # --- Modelo 2: Random Forest ---
    rf_params = train_params["random_forest"]
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params(rf_params)
        mlflow.log_param("model_type", "RandomForestClassifier")

        rf_model = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            random_state=train_params["seed"],
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        metrics = evaluate(rf_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf_model, "model")

        results["random_forest"] = metrics
        print("Random Forest:", metrics)

    # --- Modelo 3: Gradient Boosting ---
    gb_params = train_params["gradient_boosting"]
    with mlflow.start_run(run_name="gradient_boosting"):
        mlflow.log_params(gb_params)
        mlflow.log_param("model_type", "GradientBoostingClassifier")

        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params["n_estimators"],
            learning_rate=gb_params["learning_rate"],
            max_depth=gb_params["max_depth"],
            random_state=train_params["seed"],
        )
        gb_model.fit(X_train, y_train)
        metrics = evaluate(gb_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(gb_model, "model")

        results["gradient_boosting"] = metrics
        print("Gradient Boosting:", metrics)

    # --- Selección del mejor modelo según ROC-AUC ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    models_by_name = {
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "gradient_boosting": gb_model,
    }
    best_model = models_by_name[best_name]

    joblib.dump(best_model, models_dir / "best_model.pkl")

    comparison = {
        "results": results,
        "best_model": best_name,
        "best_metric_roc_auc": results[best_name]["roc_auc"],
    }
    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nMejor modelo: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")
    print(f"Guardado en {models_dir / 'best_model.pkl'}")
    print(f"Métricas comparativas en {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
