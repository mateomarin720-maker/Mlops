# actividad1-mlops-openrate-equipoXX

Mini flujo MLOps para la predicción de **Open Rate** (apertura de notificaciones push), desarrollado como Actividad Integradora 1 del módulo de Fundamentos de MLOps y Gestión del Ciclo de Vida.

> ⚠️ Antes de entregar: reemplacen `equipoXX` por su número real de equipo en el nombre del repositorio, y completen el archivo [`Integrantes_Equipo.md`](./Integrantes_Equipo.md).

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
├── docs/
│   └── screenshots/      # capturas de evidencia de MLflow
├── mlruns/               # tracking local de MLflow (NO versionado, ver sección 8.1)
├── pyproject.toml
├── poetry.lock
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── README.md
├── Integrantes_Equipo.md
└── .gitignore
```

## 4. Dataset mock

El dataset se genera de forma sintética con `src/generate_data.py` (50,000 filas por defecto) y simula relaciones realistas entre el historial de apertura, el segmento del usuario, el tipo de campaña y el horario de envío, en lugar de ruido puro. Variables incluidas:

`user_id`, `site`, `campaign_type`, `device_os`, `hour_of_day`, `day_of_week`, `historical_open_rate`, `historical_push_count`, `days_since_last_open`, `segment`, `target_opened`.

El volumen (50k filas, ~2–3 MB en CSV) justifica versionarlo con DVC en lugar de subirlo directo a Git.

## 5. Instalación y configuración del entorno

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario-u-org>/actividad1-mlops-openrate-equipoXX.git
cd actividad1-mlops-openrate-equipoXX

# 2. Instalar dependencias con Poetry (genera/actualiza poetry.lock)
poetry install

# 3. Activar el entorno virtual
poetry shell
```

> En Windows, si el comando `poetry` no se reconoce, usar `python -m poetry` en lugar de `poetry` en todos los comandos de esta guía.

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
2. **`prepare`** → `python src/prepare.py` → limpieza, encoding y feature engineering (incluye `time_bucket`, franja horaria derivada de `hour_of_day`), crea `data/processed/train.csv` y `data/processed/test.csv`
3. **`train`** → `python src/train.py` → entrena **Regresión Logística, Random Forest y Gradient Boosting**, registra las tres corridas en MLflow, guarda el mejor modelo en `models/best_model.pkl` y las métricas comparativas en `outputs/metrics.json`

El dataset generado y sus dependencias quedan versionados automáticamente por DVC como salida de la etapa `generate_data` (no requiere un `dvc add` manual). Para subir esos datos al remoto configurado:

```bash
poetry run dvc push
git add dvc.lock
git commit -m "data: versionar pipeline reproducido"
```

## 8. Visualizar los experimentos en MLflow

```bash
poetry run mlflow ui --backend-store-uri mlruns
```

Abrir `http://localhost:5000` en el navegador. Ahí se comparan, para cada corrida (`logistic_regression`, `random_forest` y `gradient_boosting`): parámetros del modelo, y métricas `accuracy`, `precision`, `recall`, `f1_score` y `roc_auc`.

### 8.1 Evidencia de experimentos registrados

La carpeta `mlruns/` **no se versiona con Git**: MLflow guarda rutas absolutas del sistema operativo en sus metadatos, lo que la hace no portable entre distintas máquinas (cada integrante genera la suya al correr `dvc repro` localmente; intentar compartirla por Git produce errores de permisos/rutas en otros equipos). En su lugar, la evidencia de los experimentos queda documentada de dos formas:

1. **`outputs/metrics.json`** (sí versionado): comparación numérica completa de los tres modelos entrenados en la última ejecución del pipeline.
2. **Capturas de pantalla** de la interfaz de MLflow, incluidas a continuación:

![Comparación de corridas en MLflow](docs/screenshots/mlflow_comparacion.png)

![Detalle de una corrida individual](docs/screenshots/mlflow_run_detalle.png)

Para explorar los experimentos de forma interactiva, cualquier integrante puede reproducir el pipeline y levantar la interfaz localmente con los dos comandos de la sección 8.

## 9. Selección del modelo

`src/train.py` selecciona automáticamente como modelo final el que obtiene mayor **ROC-AUC** sobre el conjunto de test entre los tres candidatos (Regresión Logística, Random Forest, Gradient Boosting), y lo serializa en `models/best_model.pkl`. La comparación completa queda documentada en `outputs/metrics.json`.

> Nota: al tratarse de un dataset sintético con una señal moderada y ruido intencional, se espera un desempeño modesto (ROC-AUC en el rango de 0.55–0.70 aprox.). El objetivo de la actividad es la trazabilidad del flujo, no maximizar la métrica.

## 10. Flujo de trabajo en Git/GitHub

- La rama `main` está protegida: todo cambio se integra mediante **Pull Request**, nunca con push directo.
- Ramas de trabajo utilizadas:
  - `feature/data-preparation`
  - `feature/model-training`
- Cada Pull Request debe tener al menos una revisión por parte de quien administra el repositorio antes de integrarse.
- Los commits deben ser descriptivos y reflejar el avance real del trabajo (evitar un único commit final que concentre todo).

Ejemplo de flujo para una nueva funcionalidad:

```bash
git checkout -b feature/model-training
# ... trabajo ...
git add src/train.py params.yaml dvc.yaml dvc.lock outputs/metrics.json
git commit -m "feat: agregar tercer modelo GradientBoostingClassifier y comparar contra los otros dos"
git push origin feature/model-training
# Abrir Pull Request hacia main en GitHub
```

## 11. Roles del equipo

Ver [`Integrantes_Equipo.md`](./Integrantes_Equipo.md) para el detalle de integrantes, roles y responsabilidades.

## 12. Próximos pasos / posibles extensiones

- Incorporar validación cruzada y búsqueda de hiperparámetros (`GridSearchCV`, `Optuna`).
- Agregar un notebook de EDA en `notebooks/`.
- Servir el modelo ganador con una API simple (FastAPI) como extensión del flujo MLOps.
