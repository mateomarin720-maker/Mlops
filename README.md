desarrollado como Actividad Integradora 1 del módulo de Fundamentos de MLOps y Gestión del Ciclo de Vida.


## 1. Contexto del problema

Una empresa de notificaciones digitales quiere predecir si un usuario abrirá o no una notificación push (`target_opened`), a partir de variables de comportamiento histórico, contexto temporal y segmentación del usuario. Este repositorio implementa un flujo reproducible y trazable — no solo un modelo — que cubre gestión de entorno, versionamiento de datos, entrenamiento comparativo y seguimiento de experimentos.

## 2. Stack utilizado

| Herramienta | Propósito |
|---|---|
| **Poetry** | Gestión de dependencias y entorno virtual |
| **DVC** | Versionamiento de datos y definición del pipeline reproducible |
| **MLflow** | Registro y comparación de experimentos |
| **scikit-learn** | Entrenamiento de modelos de clasificación |
| **Git / GitHub** | Control de versiones y colaboración (ramas + Pull Requests) |

## 3. Estructura del repositorio

```
actividad1-mlops-openrate-equipoXX/
├── data/
│   ├── raw/            # dataset crudo (versionado con DVC, no con Git)
│   └── processed/      # train.csv / test.csv generados por el pipeline
├── src/
│   ├── generate_data.py   # genera el dataset mock
│   ├── prepare.py         # limpieza, encoding y split train/test
│   └── train.py            # entrena y compara modelos, registra en MLflow
├── models/              # modelo ganador serializado (.pkl)
├── outputs/             # métricas comparativas (metrics.json)
├── notebooks/           # exploración (EDA), opcional
├── mlruns/               # tracking local de MLflow
├── pyproject.toml
├── poetry.lock
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── README.md
├── TEAM.md
└── .gitignore
```

## 4. Dataset mock

El dataset se genera de forma sintética con `src/generate_data.py` (50,000 filas por defecto) y simula relaciones realistas entre el historial de apertura, el segmento del usuario, el tipo de campaña y el horario de envío, en lugar de ruido puro. Variables incluidas:

`user_id`, `site`, `campaign_type`, `device_os`, `hour_of_day`, `day_of_week`, `historical_open_rate`, `historical_push_count`, `days_since_last_open`, `segment`, `target_opened`.

El volumen (50k filas, ~2–3 MB en CSV) justifica versionarlo con DVC en lugar de subirlo directo a Git.



# 2. Instalar dependencias con Poetry (genera/actualiza poetry.lock)
poetry install

# 3. Activar el entorno virtual
poetry shell
```

## 6. Inicialización de DVC (solo la primera vez, lo hace quien administra el repo)

```bash
poetry run dvc init
git add .dvc .dvcignore
git commit -m "chore: inicializar DVC"

# Remoto de almacenamiento para los datos versionados.
# Puede ser una carpeta local, Google Drive, S3, etc. Ejemplo con carpeta local:
poetry run dvc remote add -d storage ../dvc-storage-equipoXX
git add .dvc/config
git commit -m "chore: configurar remoto de DVC"
```

## 7. Reproducir el pipeline completo

El pipeline tiene tres etapas encadenadas: generación de datos → preparación → entrenamiento. Con DVC, una sola línea reproduce todo respetando las dependencias definidas en `dvc.yaml`:

```bash
poetry run dvc repro
```

Esto ejecuta en orden:

1. **`generate_data`** → `python src/generate_data.py` → crea `data/raw/open_rate_dataset.csv`
2. **`prepare`** → `python src/prepare.py` → crea `data/processed/train.csv` y `data/processed/test.csv`
3. **`train`** → `python src/train.py` → entrena Regresión Logística y Random Forest, registra ambos en MLflow, guarda el mejor modelo en `models/best_model.pkl` y las métricas comparativas en `outputs/metrics.json`

Para versionar los datos y artefactos generados:

```bash
poetry run dvc add data/raw/open_rate_dataset.csv
poetry run dvc push          # sube los datos al remoto configurado
git add data/raw/open_rate_dataset.csv.dvc dvc.lock
git commit -m "data: versionar dataset generado"
```

## 8. Visualizar los experimentos en MLflow

```bash
poetry run mlflow ui --backend-store-uri mlruns
```

Abrir `http://localhost:5000` en el navegador. Ahí se comparan, para cada corrida (`logistic_regression` y `random_forest`): parámetros del modelo, y métricas `accuracy`, `precision`, `recall`, `f1_score` y `roc_auc`.

## 9. Selección del modelo

`src/train.py` selecciona automáticamente como modelo final el que obtiene mayor **ROC-AUC** sobre el conjunto de test, y lo serializa en `models/best_model.pkl`. La comparación completa queda documentada en `outputs/metrics.json`.

> Nota: al tratarse de un dataset sintético con una señal moderada y ruido intencional, se espera un desempeño modesto (ROC-AUC en el rango de 0.55–0.70 aprox.). El objetivo de la actividad es la trazabilidad del flujo, no maximizar la métrica.

## 10. Flujo de trabajo en Git/GitHub

- La rama `main` está protegida: todo cambio se integra mediante **Pull Request**, nunca con push directo.
- Ramas de trabajo sugeridas:
  - `feature/data-preparation`
  - `feature/model-training`
  - `feature/mlflow-tracking`
- Cada Pull Request debe tener al menos una revisión por parte de quien administra el repositorio antes de integrarse.
- Los commits deben ser descriptivos y reflejar el avance real del trabajo (evitar un único commit final que concentre todo).

Ejemplo de flujo para una nueva funcionalidad:

```bash
git checkout -b feature/model-training
# ... trabajo ...
git add src/train.py
git commit -m "feat: entrenar y comparar Logistic Regression y Random Forest"
git push origin feature/model-training
# Abrir Pull Request hacia main en GitHub
```

## 11. Roles del equipo

Ver [`TEAM.md`](./TEAM.md) para el detalle de integrantes, roles y responsabilidades.

## 12. Próximos pasos / posibles extensiones

- Incorporar validación cruzada y búsqueda de hiperparámetros (`GridSearchCV`, `Optuna`).
- Agregar un notebook de EDA en `notebooks/`.
- Servir el modelo ganador con una API simple (FastAPI) como extensión del flujo MLOps.
