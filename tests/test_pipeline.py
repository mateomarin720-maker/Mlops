"""
Pruebas básicas del pipeline. No son un requisito explícito de la actividad,
pero refuerzan la evidencia de buenas prácticas de ingeniería.

Uso:
    poetry run pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import generate_dataset  # noqa: E402


def test_generate_dataset_shape():
    df = generate_dataset(n_rows=500, seed=1)
    assert len(df) == 500
    assert "target_opened" in df.columns


def test_generate_dataset_target_is_binary():
    df = generate_dataset(n_rows=500, seed=1)
    assert set(df["target_opened"].unique()).issubset({0, 1})


def test_generate_dataset_open_rate_in_reasonable_range():
    df = generate_dataset(n_rows=5000, seed=1)
    open_rate = df["target_opened"].mean()
    assert 0.05 < open_rate < 0.95
