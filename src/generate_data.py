"""
generate_data.py
-----------------
Genera un dataset mock realista para el problema de predicción de Open Rate
de notificaciones push. Este script simula datos de una plataforma de
notificaciones digitales con relaciones no triviales entre variables,
de forma que los modelos entrenados posteriormente tengan una tarea real
de aprendizaje (no un patrón trivial ni puro ruido).

Uso:
    poetry run python src/generate_data.py --n_rows 50000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SITES = ["site_co", "site_mx", "site_ar", "site_cl", "site_pe"]
CAMPAIGN_TYPES = ["promocional", "transaccional", "informativa", "reactivacion"]
DEVICE_OS = ["android", "ios", "web"]
SEGMENTS = ["nuevo", "activo", "en_riesgo", "vip", "dormido"]


def generate_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    user_id = np.arange(1, n_rows + 1)
    site = rng.choice(SITES, size=n_rows)
    campaign_type = rng.choice(
        CAMPAIGN_TYPES, size=n_rows, p=[0.4, 0.25, 0.2, 0.15]
    )
    device_os = rng.choice(DEVICE_OS, size=n_rows, p=[0.55, 0.35, 0.10])
    hour_of_day = rng.integers(0, 24, size=n_rows)
    day_of_week = rng.integers(0, 7, size=n_rows)
    segment = rng.choice(
        SEGMENTS, size=n_rows, p=[0.2, 0.35, 0.2, 0.1, 0.15]
    )

    historical_open_rate = np.clip(rng.beta(2, 5, size=n_rows), 0, 1)
    historical_push_count = rng.poisson(lam=25, size=n_rows)
    days_since_last_open = rng.exponential(scale=10, size=n_rows).astype(int)

    # --- Construcción de una probabilidad de apertura "realista" ---
    # combina efectos de segmento, horario, historial y tipo de campaña
    segment_effect = pd.Series(segment).map(
        {"vip": 0.35, "activo": 0.15, "nuevo": 0.05, "en_riesgo": -0.10, "dormido": -0.30}
    ).values

    campaign_effect = pd.Series(campaign_type).map(
        {"transaccional": 0.25, "reactivacion": -0.05, "promocional": 0.0, "informativa": -0.05}
    ).values

    hour_effect = np.where((hour_of_day >= 8) & (hour_of_day <= 21), 0.10, -0.15)
    recency_effect = -0.01 * np.clip(days_since_last_open, 0, 60)
    history_effect = 0.6 * historical_open_rate

    logit = (
        -0.7
        + segment_effect
        + campaign_effect
        + hour_effect
        + recency_effect
        + history_effect
        + rng.normal(0, 0.3, size=n_rows)  # ruido
    )
    prob_open = 1 / (1 + np.exp(-logit))
    target_opened = rng.binomial(1, prob_open)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "site": site,
            "campaign_type": campaign_type,
            "device_os": device_os,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "historical_open_rate": historical_open_rate.round(4),
            "historical_push_count": historical_push_count,
            "days_since_last_open": days_since_last_open,
            "segment": segment,
            "target_opened": target_opened,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Genera dataset mock de Open Rate")
    parser.add_argument("--n_rows", type=int, default=50000, help="Numero de filas a generar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--output", type=str, default="data/raw/open_rate_dataset.csv",
        help="Ruta de salida del CSV"
    )
    args = parser.parse_args()

    df = generate_dataset(args.n_rows, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset generado con {len(df)} filas -> {output_path}")
    print(f"Tasa de apertura global: {df['target_opened'].mean():.3f}")


if __name__ == "__main__":
    main()
