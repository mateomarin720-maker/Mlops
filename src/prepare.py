"""
prepare.py
----------
Etapa de preparación de datos del pipeline DVC.
Lee el dataset crudo, aplica limpieza básica, codifica variables
categóricas y separa en train/test, guardando los artefactos
procesados en data/processed/.

Uso:
    poetry run python src/prepare.py
"""

import yaml
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    prep_params = params["prepare"]

    raw_path = Path(prep_params["input_path"])
    processed_dir = Path(prep_params["output_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    # Limpieza básica
    df = df.drop_duplicates(subset="user_id")
    df = df.dropna()

    # One-hot encoding de variables categóricas
    categorical_cols = ["site", "campaign_type", "device_os", "segment"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # user_id no es una variable predictiva
    feature_cols = [c for c in df_encoded.columns if c not in ("user_id", "target_opened")]

    X = df_encoded[feature_cols]
    y = df_encoded["target_opened"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=prep_params["test_size"],
        random_state=prep_params["seed"],
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["target_opened"] = y_train
    test_df = X_test.copy()
    test_df["target_opened"] = y_test

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Guardado en {processed_dir}/train.csv y {processed_dir}/test.csv")


if __name__ == "__main__":
    main()
