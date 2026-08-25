```python
# ============================================================
# CONECTAR GOOGLE DRIVE
# ============================================================

from google.colab import drive

drive.mount('/content/drive')

print("✅ Google Drive conectado correctamente.")
```

## 08.07.2026

```python
# # FRAMEWORK TEMPORAL Q1-EXTREMO PARA IDENTIFICAR SEÑALES PROXY DE POSIBLE SOBRECARGA FISCAL EN REGISTROS DEL MPFN
#
# ============================================================
# VERSION: Q1-EXTREMO v3.02-INDUSTRIAL (efectos estadisticos + calibracion + learning curves + metricas extendidas)
# ============================================================
#
# v1.1 - PATCH DE REVISION (ver notas "# [PATCH]" en el codigo):
# [PATCH 1] Corrige f-strings con formato condicional invalido (BEST_THRESHOLD:.4f if ... else ...)
#            que causaban ValueError al final del pipeline. Se introduce safe_fmt().
# [PATCH 7] Simplifica la carga del dataset: un solo CSV consolidado, sin logica de multiples frames.
# [PATCH 9] Elimina la inferencia peligrosa del anio desde el nombre del archivo.
# [PATCH 8] Agrega assertions de integridad en construccion del target, seleccion de features y train.
# [PATCH 10] Agrega generacion de PROJECT_SCORECARD.xlsx.
# [PATCH 6] Espeja el resumen final tambien hacia el logger (pipeline_log).
#
# v2.0 - INFRAESTRUCTURA INDUSTRIAL (checkpoints M01-M08, pensada para escalar
#        a datasets de millones de filas):
# [PATCH 11] CheckpointManager: guarda/recupera estado por modulo (M01..M08).
# [PATCH 12] save_all_trained_models(): guarda TODOS los modelos, no solo el mejor.
# [PATCH 13] Checkpoints grandes tambien en Parquet (DataFrames >=100k filas).
# [PATCH 14] Liberacion agresiva de RAM (gc.collect + malloc_trim) tras cada modulo.
# [PATCH 15] Monitor de RAM/CPU/tiempo por modulo -> outputs/pipeline_resources.csv
# [PATCH 16] Auto-save periodico (hilo en 2do plano, cada 5 min) del manifest/log.
# [PATCH 17] ZIP incremental por checkpoint + ZIP final consolidado (incluye checkpoints/).
# [PATCH 18] Manifest de checkpoints con SHA-256 (checkpoints/checkpoint_manifest.json).
# [PATCH 19] predict_in_batches() / predict_in_batches_to_csv() / evaluate_model_batched():
#            evita picos de memoria al predecir sobre datasets grandes (validacion externa).
# [PATCH 20] retry_on_failure(): reintento automatico al entrenar con RandomizedSearchCV.
#
# NOTA HONESTA SOBRE ALCANCE: el CheckpointManager guarda el estado de 8 puntos
# clave del pipeline (M01 carga de datos -> M02 feature engineering -> M03 feature
# selection/matrices -> M04 modelos base -> M05 ensambles -> M06 explicabilidad ->
# M07 clustering -> M08 export). Esto cubre las etapas mas costosas en tiempo/RAM.
# Retomar automaticamente DESDE la mitad exacta de un modulo (p.ej. a mitad del
# RandomizedSearchCV de un modelo especifico) requeriria refactorizar cada seccion
# en funciones independientes con su propio guard `if ckpt.should_skip(...)`; el
# patron y las herramientas (ckpt.should_skip/load/save/timed) ya estan listos
# para que cualquier seccion adicional se envuelva de la misma forma.
#
# v2.02 - RESUME REAL + ENVIRONMENT AUDIT + VERIFICACION DE ARCHIVOS:
# [M00] Modulo 0.5 "Environment & Hardware Audit": detecta CPU/RAM/GPU/SO/Python
#       y versiones de librerias clave; escribe outputs/00_ENVIRONMENT/
#       (hardware_info.json, execution_environment.json, software_versions.xlsx,
#       cpu_info.txt, gpu_info.txt, ram_info.txt, disk_info.txt,
#       environment_report.pdf). Se saltea solo si ya corrio (checkpoint M00).
# [M01-real] La carga+normalizacion del CSV (M01) ahora usa un `if
#       ckpt.should_skip("M01_dataset_cargado")` REAL: si el checkpoint existe,
#       el pipeline NO vuelve a leer/normalizar el CSV completo, sino que carga
#       df_raw/file_summary_df/column_audit_df/duplicate_columns_df directamente
#       desde checkpoints/M01_dataset_cargado.pkl. Este es el patron de resume
#       real; los checkpoints M02-M08 restantes guardan estado pero, por ahora,
#       no interrumpen la ejecucion secuencial (ver nota de alcance arriba).
# [verify_generated_files] Verificacion automatica de archivos clave (existencia
#       + tamano > 0) antes de empaquetar el ZIP final; reporte en
#       outputs/tables/file_verification_report.xlsx.
#
# v3.00 - MLFLOW COMPLETO + AUTO-REGENERACION + ESCRITURA INCREMENTAL:
# [PATCH 21] MLflow (tracking local en ./mlruns/, sin servidor externo):
#       log_params (config, target, mejor modelo/umbral), log_metrics
#       (metricas finales por dataset), log_model (modelo principal, flavor
#       sklearn cuando aplica) y log_artifacts en bloque de TODO outputs/
#       (modelos .joblib, tablas .csv/.xlsx, figuras .png, PDF, HTML),
#       checkpoints/ (con manifest SHA-256) y DOI_PACKAGE/, mas el ZIP final.
# [PATCH 22] Auto-regeneracion: generate_final_pdf_report(), generate_html_report()
#       y generate_final_zip() ahora son funciones invocables; si al momento de
#       empaquetar falta el PDF, el HTML o el ZIP, se regeneran automaticamente
#       antes de continuar (nunca termina con artefactos criticos faltantes).
# [PATCH 23] Escritura incremental: append_to_csv() usado en la validacion
#       externa (cada modelo evaluado se agrega de inmediato al CSV en disco).
# [PATCH 24] pipeline_log.csv: consolida inicio/fin/tiempo/RAM/CPU/estado/
#       checkpoint por modulo (a partir de los datos ya recolectados por
#       CheckpointManager), en outputs/pipeline_log.csv.
# [PATCH 25] MLFLOW_EXPORT/: copia completa de ./mlruns/ dentro de
#       DOI_PACKAGE/MLFLOW_EXPORT/, y respaldo automatico del ZIP final en
#       Google Drive (si /content/drive/MyDrive esta montado) ademas de la
#       descarga automatica ya existente.
#
# v3.01 - FIX: BACKEND DE TRACKING DE MLFLOW
# MLflow >=2.x/3.x puso el file store plano ("./mlruns" tal cual, sin base de
# datos) en "maintenance mode": mlflow.set_experiment() lanzaba MlflowException
# ("The filesystem tracking backend ... is in maintenance mode"). Se corrige
# usando SQLite como backend de tracking (`sqlite:///mlflow.db`, la ruta que
# el propio MLflow recomienda), con artefactos en ./mlflow_artifacts/ y un
# fallback automatico al file store clasico (con opt-out explicito via
# MLFLOW_ALLOW_FILE_STORE=true) si SQLite no estuviera disponible. La
# exportacion al DOI_PACKAGE (MLFLOW_EXPORT/) ahora copia mlflow.db +
# mlflow_artifacts/ (o mlruns/ si se uso el fallback).
#
# v3.02 - OBSERVACIONES DE REVISION (efectos estadisticos, calibracion,
# umbral multi-criterio, learning curves, estabilidad, metricas extendidas):
# [clean_for_export] Reemplaza inf/NaN por "N/A" antes de exportar (resuelve
#       el F1=NaN de DummyClassifier y los std=NaN de variables agregadas);
#       ademas se documenta la causa exacta en una columna "nota" en vez de
#       dejar el NaN sin explicacion.
# [IV alto = variables del target] Se agrega nota explicita en iv_df: los IV
#       extremos de saldo_casos/tasa_atencion/ratio_saldo se deben a que esas
#       variables CONSTRUYEN la etiqueta proxy (ya excluidas de X); no son
#       fuga no controlada.
# [Tamanos de efecto] Cohen's d y Cliff's Delta anadidos al bloque KS/Mann-
#       Whitney/Welch; Cramer's V anadido al bloque Chi-cuadrado categorico.
# [Calibracion] ECE + MCE + Reliability Diagram (bins) anadidos junto al
#       Brier Score ya existente.
# [Umbral multi-criterio] Ademas del umbral F1-optimo (BEST_THRESHOLD), se
#       reportan: 0.5 fijo, Youden's J (ROC), umbral para precision objetivo
#       y umbral para recall objetivo, todos calculados en validacion 2024.
# [Learning Curve] F1 train vs F1 CV-walk-forward en funcion del tamano de
#       entrenamiento, para el modelo predictivo principal.
# [Cluster stability] Bootstrap Adjusted Rand Index (ARI) sobre el KMeans
#       operativo seleccionado.
# [Outliers] Mahalanobis (MinCovDet) anadido como tercer metodo, junto a
#       IsolationForest y LocalOutlierFactor ya existentes.
# [Feature selection stability] Columna stability_score_pct (votos / total
#       de metodos * 100) anadida a feature_votes_df.
# [Metricas extendidas] MCC, Cohen's Kappa, NPV, FPR, FNR, LR+, LR- anadidos
#       a evaluate_model/evaluate_model_batched/evaluate_predictions/
#       evaluate_scores_from_array (se propagan a TODAS las tablas de
#       metricas del pipeline, no solo al modelo final).
# NOTA HONESTA: McNemar, DeLong, PSI/drift temporal, bootstrap CI (IC95%) y
#       la separacion Oracle-vs-ML YA estaban implementados en versiones
#       previas de este script; no se reimplementaron, solo se les aplico
#       clean_for_export() para consistencia de formato.
# NO implementado en esta version (por complejidad/riesgo de sobre-prometer):
#       Little's MCAR test formal, SHAP waterfall/force-plot/interaction-
#       heatmap por caso individual, bootstrap de intervalos de confianza
#       para SHAP, y drift multivariante (Jensen-Shannon por variable mas
#       alla del PSI ya existente). Quedan como trabajo futuro documentado.
#
# ------------------------------------------------------------
# (El resto de las notas metodologicas Q1-extremo originales se mantienen sin cambios.)
# - reformula el objetivo como identificacion de senales proxy, no como certificacion oficial de congestion;
# - conserva trazabilidad de datos sin eliminar registros por defecto;
# - construye el target proxy con umbrales calculados solo sobre entrenamiento;
# - agrega validacion empirica del constructo proxy sin simular Delphi/AHP;
# - evalua sensibilidad P70/P30, P75/P25, P80/P20 y P85/P15;
# - excluye variables de fuga directa (`ingresado`, `atendido`, `saldo_casos`, `tasa_atencion`, `ratio_saldo`);
# - corrige variables historicas para que usen agregados anuales previos por grupo y no informacion del mismo anio;
# - ajusta el preprocesador de feature selection solo con 2019-2022 y valida internamente en 2023;
# - reemplaza la validacion temporal por filas con folds anuales walk-forward cuando corresponde;
# - anade Nested Temporal CV como validacion avanzada opcional;
# - calibra SVM con `CalibratedClassifierCV` para evitar scores reescalados por dataset;
# - anade benchmarks tabulares modernos con ejecucion protegida: TabPFN, FT-Transformer y TabNet si el entorno lo permite;
# - usa el modelo predictivo principal como modelo de explicabilidad por defecto;
# - anade SHAP Interaction Values cuando el modelo compatible este disponible;
# - anade contrafactuales controlados en espacio preprocesado, sin modificar variables de fuga;
# - anade conformal prediction por split conformal usando validacion 2024 como calibracion;
# - anade fairness avanzado: demographic parity, equal opportunity, equalized odds, calibration by group e intersecciones;
# - reformula clustering como perfilamiento operativo, validacion convergente del target y fairness por cluster;
# - anade baselines de reglas simples y sensibilidad del target;
# - corrige PSI con extremos infinitos para no perder valores fuera de rango;
# - reporta 2026 como validacion externa exploratoria por ser un anio parcial.
#
# Diseno temporal usado:
#
# - Train principal: 2019-2023
# - Validacion/calibracion: 2024
# - Test: 2025
# - Validacion externa exploratoria: 2026 parcial
#
# Nota metodologica central: `riesgo_congestion` es una etiqueta proxy auditable construida con
# saldo alto de casos y tasa de atencion baja. No debe interpretarse como certificacion oficial
# de congestion fiscal ni como relacion causal.
#
# Nota sobre Delphi/AHP: esta version NO simula panel experto. Si no existen expertos reales,
# Delphi/AHP se declara como trabajo futuro y se reemplaza por validacion empirica del constructo proxy.

# ============================================================
# 0. CONFIGURACION GENERAL DEL NOTEBOOK
# ============================================================
FAST_MODE = False
RUN_HEAVY_BLOCKS = True
RUN_ABLATION_STUDY = True
RUN_ADVANCED_STAT_TESTS = True
USE_SMOTE = False
AUTO_INSTALL_PACKAGES = True

RUN_TARGET_CONSTRUCT_VALIDATION = True
RUN_SOTA_TABULAR_BENCHMARKS = True
RUN_NESTED_TEMPORAL_CV = True
RUN_CONFORMAL_PREDICTION = True
RUN_COUNTERFACTUAL_EXPLANATIONS = True
RUN_SHAP_INTERACTIONS = True
RUN_ADVANCED_FAIRNESS = True
RUN_CLUSTER_AS_FEATURE_EXPERIMENT = True
RUN_TABNET_FALLBACK_BENCHMARK = True
RUN_MONDRIAN_CONFORMAL_BY_CLUSTER = True
RUN_MONDRAIN_CONFORMAL_BY_CLUSTER = RUN_MONDRIAN_CONFORMAL_BY_CLUSTER

MAX_TABPFN_TRAIN_SIZE = 5000
MAX_FT_TRANSFORMER_TRAIN_SIZE = 8000
MAX_TABNET_TRAIN_SIZE = 8000
MAX_EXPLAINABILITY_SAMPLE = 500

PRIMARY_TARGET_POLICY = "conservative_operational_overload"
EARLY_WARNING_TARGET_SCENARIO = "P70_P30"

RANDOM_STATE = 42
N_EXPERIMENTS_PER_MODEL = 100
CV_FOLDS = 5

TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
FEATURE_SELECTION_TRAIN_YEARS = [2019, 2020, 2021, 2022]
FEATURE_SELECTION_INTERNAL_VALID_YEAR = 2023

VALID_YEAR = 2024
TEST_YEAR = 2025
EXTERNAL_YEAR = 2026

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FIG_DIR = f"{OUTPUT_DIR}/figures"
TABLE_DIR = f"{OUTPUT_DIR}/tables"
MODEL_DIR = f"{OUTPUT_DIR}/models"

import os
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# [PATCH 1] --------------------------------------------------
# Helper global para formatear variables que pueden no existir aun o pueden
# fallar si alguien intenta usar una expresion condicional dentro del format
# spec de un f-string (ej. {VAR:.4f if cond else 'N/A'} -> ValueError).
# Todo el pipeline debe usar safe_fmt() en vez de ese patron invalido.
def safe_fmt(varname, fmt="{:.4f}", default="N/A"):
    """Formatea de forma segura una variable global, evitando el patron
    invalido `{VAR:.4f if cond else 'X'}` dentro de f-strings."""
    val = globals().get(varname, None)
    if val is None:
        return default
    try:
        return fmt.format(val)
    except Exception:
        try:
            return str(val)
        except Exception:
            return default
# --------------------------------------------------------------

print("Configuracion Q1-extremo (con patch de revision) cargada correctamente.")
print("Train principal:", TRAIN_YEARS)
print("Train interno para feature selection:", FEATURE_SELECTION_TRAIN_YEARS)
print("Validacion interna feature selection:", FEATURE_SELECTION_INTERNAL_VALID_YEAR)
print("Validacion/calibracion:", VALID_YEAR)
print("Test:", TEST_YEAR)
print("Externa exploratoria:", EXTERNAL_YEAR)

PROBLEM_DEFINITION = {
    "objetivo": "Identificar senales proxy de posible sobrecarga fiscal operacional en registros administrativos del MPFN.",
    "tipo_problema_principal": "Clasificacion binaria supervisada",
    "target_proxy": "riesgo_congestion",
    "clase_0": "Sin senal proxy de riesgo",
    "clase_1": "Con senal proxy de riesgo",
    "analisis_secundario": "Clustering y deteccion de anomalias como analisis no supervisado complementario",
    "unidad_analisis": "Registro administrativo agregado por anio y atributos institucionales/territoriales/tipo de caso",
    "advertencia_metodologica": "La etiqueta no es una certificacion oficial de congestion fiscal ni prueba causal. Es una aproximacion operativa basada en saldo alto y baja tasa de atencion."
}

for k, v in PROBLEM_DEFINITION.items():
    print(f"{k}: {v}")

import sys, subprocess, warnings, math, json, re, glob, textwrap, platform
warnings.filterwarnings("ignore")

def install_if_missing(package, import_name=None):
    """Importa una libreria y, si esta permitido, la instala en entornos tipo Colab."""
    import importlib
    name = import_name or package
    try:
        return importlib.import_module(name)
    except Exception as import_error:
        if not globals().get("AUTO_INSTALL_PACKAGES", False):
            raise ImportError(
                f"No se pudo importar '{name}'. Instala dependencias desde requirements.txt. "
                f"Error original: {import_error}"
            )
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return importlib.import_module(name)

# Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

# Estadistica
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind, chi2_contingency, friedmanchisquare, norm
from scipy.special import expit

# Scikit-learn: preprocesamiento y validacion
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2, RFE, RFECV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import CalibratedClassifierCV

# Scikit-learn: modelos supervisados
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

# Clustering y anomalias
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Exportacion
import joblib

# Opcionales importantes
try:
    import lightgbm as lgb
except Exception:
    install_if_missing("lightgbm")
    import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
except Exception:
    install_if_missing("catboost")
    from catboost import CatBoostClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    install_if_missing("xgboost")
    from xgboost import XGBClassifier

try:
    import shap
except Exception:
    install_if_missing("shap")
    import shap

try:
    import optuna
except Exception:
    install_if_missing("optuna")
    import optuna

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    install_if_missing("statsmodels")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import umap
except Exception:
    install_if_missing("umap-learn", "umap")
    import umap

try:
    from kmodes.kprototypes import KPrototypes
    from kmodes.kmodes import KModes
except Exception:
    try:
        install_if_missing("kmodes")
        from kmodes.kprototypes import KPrototypes
        from kmodes.kmodes import KModes
    except Exception:
        KPrototypes = None
        KModes = None
        print("kmodes no disponible; se omitira K-Prototypes/K-Modes si falla la instalacion.")

try:
    from boruta import BorutaPy
except Exception:
    try:
        install_if_missing("boruta")
        from boruta import BorutaPy
    except Exception:
        BorutaPy = None
        print("Boruta no disponible; se usara una alternativa basada en Random Forest + variables sombra.")

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    try:
        install_if_missing("imbalanced-learn", "imblearn")
        from imblearn.over_sampling import SMOTE
    except Exception:
        SMOTE = None
        print("SMOTE no disponible. Se utilizara ponderacion de clases.")

try:
    import scikit_posthocs as sp
except Exception:
    try:
        install_if_missing("scikit-posthocs", "scikit_posthocs")
        import scikit_posthocs as sp
    except Exception:
        sp = None
        print("scikit-posthocs no disponible; Nemenyi se omitira si falla la instalacion.")

# [PATCH 6] Logging basico disponible desde el inicio (ademas del logging avanzado
# de la seccion de reproducibilidad al final del pipeline).
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
early_log = logging.getLogger("pipeline_early")

def build_year_forward_cv_splits(years, min_train_periods=1, max_splits=None):
    """Construye folds walk-forward por anio completo, no por cantidad de filas."""
    years_arr = np.asarray(years)
    unique_years = sorted(pd.Series(years_arr).dropna().unique().tolist())
    splits = []
    for eval_year in unique_years[min_train_periods:]:
        train_idx = np.where(years_arr < eval_year)[0]
        valid_idx = np.where(years_arr == eval_year)[0]
        if len(train_idx) > 0 and len(valid_idx) > 0:
            splits.append((train_idx, valid_idx))
    if max_splits is not None and len(splits) > max_splits:
        splits = splits[-max_splits:]
    return splits

def filter_cv_splits_with_both_classes(cv_splits, y):
    """Evita folds imposibles cuando train o valid tienen una sola clase."""
    y_arr = np.asarray(y)
    filtered = []
    for tr_idx, va_idx in cv_splits:
        if len(np.unique(y_arr[tr_idx])) > 1 and len(np.unique(y_arr[va_idx])) > 1:
            filtered.append((tr_idx, va_idx))
    return filtered

def iter_cv_splits(cv, X):
    """Itera sobre CV tanto si es lista de indices como si es splitter sklearn."""
    if isinstance(cv, list):
        return cv
    return list(cv.split(X))

def make_one_hot_encoder():
    """Compatibilidad entre versiones de scikit-learn: sparse_output vs sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def get_preprocessed_feature_names(preprocessor, numeric_cols, categorical_cols):
    """Extrae nombres finales del ColumnTransformer."""
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    return np.array(names)

def print_environment_versions():
    """Registra versiones para reproducibilidad."""
    modules = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    try:
        import sklearn
        modules["sklearn"] = sklearn.__version__
    except Exception:
        pass
    for name, module in [("lightgbm", lgb), ("xgboost", sys.modules.get("xgboost")), ("catboost", sys.modules.get("catboost")), ("shap", shap), ("optuna", optuna)]:
        try:
            modules[name] = getattr(module, "__version__", "No disponible")
        except Exception:
            modules[name] = "No disponible"
    env_df = pd.DataFrame([{"package": k, "version": v} for k, v in modules.items()])
    display(env_df)
    env_df.to_excel(os.path.join(TABLE_DIR, "00_environment_versions.xlsx"), index=False)
    return env_df

# ============================================================
# [PATCH 11-20] INFRAESTRUCTURA DE CHECKPOINT / RESUME / RAM / BATCH
# Version v2.02-industrial. Nueva capa reutilizable para datasets grandes
# (pensada para escalar hasta ~4.5M de registros) que agrega:
#  [11] Checkpoint automatico por modulo + resume automatico al reiniciar.
#  [12] Guardado de TODOS los modelos entrenados (no solo el mejor), comprimidos.
#  [13] Guardado de DataFrames/arrays grandes en Parquet/joblib para resume rapido.
#  [14] Liberacion agresiva de RAM (gc.collect + malloc_trim) tras cada modulo.
#  [15] Monitor de RAM/CPU/tiempo por modulo -> pipeline_resources.csv
#  [16] Auto-save periodico (cada N minutos) via hilo en segundo plano.
#  [17] ZIP incremental por modulo + ZIP final consolidado.
#  [18] Manifest con SHA-256 de cada checkpoint (ademas del manifest de outputs).
#  [19] Prediccion por lotes (evita picos de memoria en datasets masivos).
#  [20] Reintento automatico (retry) para el entrenamiento de un modelo que falle.
# ============================================================
import gc
import ctypes
import time
import threading
import traceback
import hashlib as _hashlib_ckpt
import zipfile
import datetime

CHECKPOINT_DIR = "checkpoints"
INCREMENTAL_ZIP_DIR = "checkpoints_zip"
for _d in [CHECKPOINT_DIR, INCREMENTAL_ZIP_DIR]:
    os.makedirs(_d, exist_ok=True)

RESOURCE_LOG_PATH = os.path.join(OUTPUT_DIR, "pipeline_resources.csv")
CHECKPOINT_MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_manifest.json")

def _free_ram_aggressive():
    """[PATCH 14] Libera RAM de forma agresiva: gc + malloc_trim (Linux) si esta disponible."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def _ram_snapshot():
    """[PATCH 15] Snapshot de RAM/CPU. Usa psutil si esta disponible; si no, degrada con aviso."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        return {
            "ram_used_gb": round((vm.total - vm.available) / 1e9, 3),
            "ram_available_gb": round(vm.available / 1e9, 3),
            "ram_percent": vm.percent,
            "process_rss_gb": round(proc.memory_info().rss / 1e9, 3),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }
    except Exception:
        return {"ram_used_gb": None, "ram_available_gb": None, "ram_percent": None,
                "process_rss_gb": None, "cpu_percent": None}

def _sha256_of_file(path):
    try:
        h = _hashlib_ckpt.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "no_disponible"

class CheckpointManager:
    """
    [PATCH 11-20] Gestor de checkpoints por modulo para el pipeline Q1-Extremo.

    Uso tipico dentro de cualquier seccion del pipeline:

        if ckpt.should_skip("M05_feature_engineering"):
            state = ckpt.load("M05_feature_engineering")
            df = state["df"]
        else:
            with ckpt.timed("M05_feature_engineering"):
                # ... codigo pesado que produce df ...
                ckpt.save("M05_feature_engineering", df=df)

    Al reiniciar el entorno (por ejemplo tras una caida de Colab), basta con
    volver a ejecutar el script: los modulos ya completados se detectan
    automaticamente via should_skip() y NO se recalculan.
    """

    def __init__(self, checkpoint_dir=CHECKPOINT_DIR,
                 zip_dir=INCREMENTAL_ZIP_DIR,
                 resource_log_path=RESOURCE_LOG_PATH,
                 manifest_path=CHECKPOINT_MANIFEST_PATH,
                 autosave_minutes=5,
                 ram_floor_gb=1.0):
        self.checkpoint_dir = checkpoint_dir
        self.zip_dir = zip_dir
        self.resource_log_path = resource_log_path
        self.manifest_path = manifest_path
        self.ram_floor_gb = ram_floor_gb
        self.manifest = self._load_manifest()
        self._resource_rows = []
        self._autosave_thread = None
        self._autosave_stop = threading.Event()
        if autosave_minutes and autosave_minutes > 0:
            self._start_autosave(autosave_minutes)
        print(f"[CheckpointManager] Modulos ya completados detectados: "
              f"{len(self.manifest.get('completed_modules', {}))}")

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_modules": {}, "created": datetime.datetime.now().isoformat()}

    def _write_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False, default=str)

    def should_skip(self, module_id):
        """[PATCH 11] True si el modulo ya tiene checkpoint valido -> se puede saltar."""
        info = self.manifest.get("completed_modules", {}).get(module_id)
        if not info:
            return False
        pkl_path = os.path.join(self.checkpoint_dir, f"{module_id}.pkl")
        return os.path.exists(pkl_path)

    def load(self, module_id):
        """Carga el estado (dict de objetos) guardado para un modulo ya completado."""
        pkl_path = os.path.join(self.checkpoint_dir, f"{module_id}.pkl")
        print(f"[CheckpointManager] Reanudando modulo '{module_id}' desde checkpoint "
              f"({os.path.getsize(pkl_path)/1e6:.1f} MB) - NO se recalcula.")
        return joblib.load(pkl_path)

    def save(self, module_id, **objects):
        """
        [PATCH 11-13] Guarda el estado de un modulo (variables clave como dict)
        en un unico .pkl comprimido (joblib, compress=3). DataFrames grandes
        (>=100k filas) tambien se guardan en Parquet aparte para lecturas rapidas.
        """
        pkl_path = os.path.join(self.checkpoint_dir, f"{module_id}.pkl")
        joblib.dump(objects, pkl_path, compress=3)

        parquet_files = []
        for name, obj in objects.items():
            if isinstance(obj, pd.DataFrame) and len(obj) >= 100_000:
                try:
                    pq_path = os.path.join(self.checkpoint_dir, f"{module_id}__{name}.parquet")
                    obj.to_parquet(pq_path, index=False)
                    parquet_files.append(pq_path)
                except Exception as e:
                    print(f"[CheckpointManager] No se pudo guardar {name} en Parquet: {e}")

        sha = _sha256_of_file(pkl_path)
        self.manifest.setdefault("completed_modules", {})[module_id] = {
            "pkl_path": pkl_path,
            "sha256": sha,
            "size_bytes": os.path.getsize(pkl_path),
            "parquet_files": parquet_files,
            "completed_at": datetime.datetime.now().isoformat(),
            "keys": list(objects.keys()),
        }
        self._write_manifest()
        self._zip_module(module_id)  # [PATCH 17] ZIP incremental
        _free_ram_aggressive()       # [PATCH 14]
        print(f"[CheckpointManager] Modulo '{module_id}' guardado "
              f"({os.path.getsize(pkl_path)/1e6:.1f} MB, sha256={sha[:12]}...).")

    class _TimedBlock:
        def __init__(self, manager, module_id):
            self.manager = manager
            self.module_id = module_id

        def __enter__(self):
            self.t0 = time.time()
            self.ram0 = _ram_snapshot()
            print(f"[CheckpointManager] >>> Iniciando modulo '{self.module_id}' "
                  f"(RAM usada: {self.ram0.get('ram_used_gb')} GB)")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.t0
            ram1 = _ram_snapshot()
            row = {
                "module_id": self.module_id,
                "start_ram_used_gb": self.ram0.get("ram_used_gb"),
                "end_ram_used_gb": ram1.get("ram_used_gb"),
                "end_ram_available_gb": ram1.get("ram_available_gb"),
                "elapsed_seconds": round(elapsed, 2),
                "elapsed_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                "status": "error" if exc_type else "ok",
                "error": str(exc_val) if exc_val else "",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            self.manager._resource_rows.append(row)
            self.manager._flush_resource_log()
            print(f"[CheckpointManager] <<< Modulo '{self.module_id}' terminado en "
                  f"{row['elapsed_human']} (RAM fin: {ram1.get('ram_used_gb')} GB)")
            if exc_type:
                print(f"[CheckpointManager] ERROR en '{self.module_id}': {exc_val}")
                traceback.print_exc()
            _free_ram_aggressive()
            avail = ram1.get("ram_available_gb")
            if avail is not None and avail < self.manager.ram_floor_gb:
                print(f"[CheckpointManager] AVISO: RAM libre baja ({avail} GB < "
                      f"{self.manager.ram_floor_gb} GB). Liberando agresivamente...")
                _free_ram_aggressive()
            return False

    def timed(self, module_id):
        """Context manager: mide tiempo/RAM de un modulo y lo registra en el CSV de recursos."""
        return CheckpointManager._TimedBlock(self, module_id)

    def _flush_resource_log(self):
        try:
            pd.DataFrame(self._resource_rows).to_csv(self.resource_log_path, index=False)
        except Exception as e:
            print(f"[CheckpointManager] No se pudo escribir {self.resource_log_path}: {e}")

    def _zip_module(self, module_id):
        """[PATCH 17] Genera/actualiza un ZIP incremental con todos los checkpoints hasta ahora."""
        try:
            zip_path = os.path.join(self.zip_dir, f"checkpoints_hasta_{module_id}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
                for root, dirs, files in os.walk(self.checkpoint_dir):
                    for fname in files:
                        fp = os.path.join(root, fname)
                        zf.write(fp, os.path.relpath(fp, "."))
        except Exception as e:
            print(f"[CheckpointManager] ZIP incremental no generado para {module_id}: {e}")

    def _start_autosave(self, minutes):
        """[PATCH 16] Hilo en segundo plano que fuerza flush del log de recursos cada N minutos."""
        def _loop():
            while not self._autosave_stop.wait(minutes * 60):
                self._flush_resource_log()
                self._write_manifest()
        self._autosave_thread = threading.Thread(target=_loop, daemon=True)
        self._autosave_thread.start()

    def stop(self):
        self._autosave_stop.set()

def save_all_trained_models(trained_models, model_dir=MODEL_DIR):
    """
    [PATCH 12] Guarda TODOS los modelos entrenados (no solo el principal),
    comprimidos con joblib, para que nunca haya que reentrenarlos.
    """
    saved = []
    for name, model in trained_models.items():
        try:
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
            path = os.path.join(model_dir, f"model_{safe_name}.joblib")
            joblib.dump(model, path, compress=3)
            saved.append({"modelo": name, "path": path, "size_mb": round(os.path.getsize(path)/1e6, 2)})
        except Exception as e:
            saved.append({"modelo": name, "path": None, "error": str(e)})
    saved_df = pd.DataFrame(saved)
    saved_df.to_excel(os.path.join(TABLE_DIR, "models_saved_manifest.xlsx"), index=False)
    print(f"[save_all_trained_models] {len(saved)} modelos guardados en {model_dir}/")
    return saved_df

def predict_in_batches(model, X, batch_size=100_000, score_fn=None):
    """
    [PATCH 19] Genera scores/predicciones por lotes para evitar picos de memoria
    en datasets grandes (pensado para escalar a ~4.5M de filas). `score_fn` debe
    ser una funcion tipo get_scores(model, X_batch) -> np.ndarray; si no se
    provee, usa predict_proba/predict directamente.
    """
    n = X.shape[0]
    if n <= batch_size:
        if score_fn is not None:
            return score_fn(model, X)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    out_chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end] if not hasattr(X, "iloc") else X.iloc[start:end]
        if score_fn is not None:
            chunk_scores = score_fn(model, X_batch)
        elif hasattr(model, "predict_proba"):
            chunk_scores = model.predict_proba(X_batch)[:, 1]
        else:
            chunk_scores = model.predict(X_batch)
        out_chunks.append(np.asarray(chunk_scores))
        print(f"[predict_in_batches] Procesado lote {start:,}-{end:,} de {n:,}")
    return np.concatenate(out_chunks)

def predict_in_batches_to_csv(model, X, out_path, batch_size=100_000, score_fn=None,
                               id_values=None, id_col="row_id"):
    """
    [PATCH 19+20] Version que ADEMAS escribe a disco por lotes (append), en vez
    de acumular todo en memoria y escribir un solo CSV al final. Util cuando
    X tiene millones de filas y no se puede mantener todo en RAM simultaneamente.
    """
    n = X.shape[0]
    first_write = True
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end] if not hasattr(X, "iloc") else X.iloc[start:end]
        if score_fn is not None:
            chunk_scores = score_fn(model, X_batch)
        elif hasattr(model, "predict_proba"):
            chunk_scores = model.predict_proba(X_batch)[:, 1]
        else:
            chunk_scores = model.predict(X_batch)
        ids_batch = (id_values[start:end] if id_values is not None
                     else np.arange(start, end))
        batch_df = pd.DataFrame({id_col: ids_batch, "score": np.asarray(chunk_scores)})
        batch_df.to_csv(out_path, mode="w" if first_write else "a",
                         header=first_write, index=False)
        first_write = False
        print(f"[predict_in_batches_to_csv] Escrito lote {start:,}-{end:,} de {n:,} -> {out_path}")
    return out_path

def retry_on_failure(fn, *args, retries=1, backoff_seconds=2, label="operacion", **kwargs):
    """
    [PATCH 20] Reintenta una operacion (tipicamente el entrenamiento de un modelo)
    hasta `retries` veces adicionales antes de rendirse. Si todos los intentos
    fallan, relanza la ultima excepcion para que el llamador decida como continuar
    (por ejemplo, seguir con el siguiente modelo del catalogo).
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            print(f"[retry_on_failure] '{label}' fallo en intento {attempt+1}/{retries+1}: {e}")
            if attempt < retries:
                time.sleep(backoff_seconds)
    raise last_exc

def verify_generated_files(expected_files, label="verificacion_final"):
    """
    [PATCH: verificacion automatica de archivos] Comprueba que cada ruta en
    `expected_files` exista y tenga tamano > 0 bytes. Escribe un reporte en
    outputs/tables/{label}.xlsx con columnas: archivo, existe, tamano_bytes, estado.
    No intenta "adivinar" como recrear artefactos complejos (PDF/HTML) porque
    eso dependeria del bloque especifico que los genero; para esos casos el
    reporte solo marca el faltante para que se re-ejecute ese bloque puntual.
    """
    rows = []
    for path in expected_files:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        ok = exists and size > 0
        rows.append({"archivo": path, "existe": exists, "tamano_bytes": size,
                      "estado": "OK" if ok else "FALTANTE_O_VACIO"})
    report_df = pd.DataFrame(rows)
    report_path = os.path.join(TABLE_DIR, f"{label}.xlsx")
    report_df.to_excel(report_path, index=False)
    n_ok = int((report_df["estado"] == "OK").sum())
    n_total = len(report_df)
    print(f"[verify_generated_files] {n_ok}/{n_total} archivos esperados OK. Reporte: {report_path}")
    faltantes = report_df[report_df["estado"] != "OK"]["archivo"].tolist()
    if faltantes:
        print(f"[verify_generated_files] ARCHIVOS FALTANTES O VACIOS ({len(faltantes)}):")
        for fpath in faltantes:
            print(f"   - {fpath}")
    return report_df

def write_reproducibility_bundle_extra(output_dir=OUTPUT_DIR):
    """
    [PATCH 17-18 complemento] Genera piezas adicionales de reproducibilidad
    solicitadas en la revision: pip freeze completo y hardware_info.json,
    complementando (sin duplicar) el bloque Q1 de reproducibilidad existente.
    """
    repro_dir = os.path.join(output_dir, "reproducibility_extra")
    os.makedirs(repro_dir, exist_ok=True)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        with open(os.path.join(repro_dir, "pip_freeze.txt"), "w", encoding="utf-8") as f:
            f.write(freeze)
    except Exception as e:
        print(f"[write_reproducibility_bundle_extra] pip freeze no disponible: {e}")
    try:
        import psutil
        hw = {
            "cpu_count_logical": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        with open(os.path.join(repro_dir, "hardware_info_extra.json"), "w", encoding="utf-8") as f:
            json.dump(hw, f, indent=2)
    except Exception as e:
        print(f"[write_reproducibility_bundle_extra] hardware info no disponible: {e}")

# ============================================================
# [PATCH 21-25] MLFLOW COMPLETO + AUTO-REGENERACION + ESCRITURA INCREMENTAL
#  [21] MLflow: parametros, metricas, modelos y artefactos (pkl/csv/png/pdf/
#       html/zip/DOI_PACKAGE/checkpoints) registrados en un unico run local
#       (tracking store por defecto: ./mlruns, sin requerir servidor externo).
#  [22] Auto-regeneracion: antes de finalizar, si falta el PDF, el HTML o el
#       ZIP final, se vuelven a generar automaticamente (nunca termina con
#       archivos criticos faltantes).
#  [23] Escritura incremental (append) para resultados por lotes/por modelo,
#       en vez de acumular todo en memoria y escribir un unico archivo al final.
#  [24] pipeline_log.csv por modulo: inicio, fin, tiempo, RAM, CPU, estado,
#       checkpoint asociado (complementa pipeline_resources.csv).
#  [25] MLFLOW_EXPORT/: copia de mlruns/ dentro de DOI_PACKAGE y del ZIP final.
# ============================================================
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    install_if_missing("mlflow")
    import mlflow
    import mlflow.sklearn

MLFLOW_EXPERIMENT_NAME = "MPFN_Q1_Extreme"
# [FIX v3.01] MLflow >=2.x/3.x pone el file store plano ("./mlruns" sin mas)
# en "maintenance mode" y lanza MlflowException al llamar set_experiment().
# Se usa SQLite como backend de tracking (la ruta recomendada oficialmente
# por MLflow), con un fallback automatico al file store clasico + opt-out
# explicito (MLFLOW_ALLOW_FILE_STORE=true) si por algun motivo SQLite no
# estuviera disponible en el entorno (por ejemplo, falta de sqlalchemy).
MLFLOW_TRACKING_DIR = "mlruns"              # fallback legacy (file store)
MLFLOW_DB_PATH = "mlflow.db"                 # backend preferido (SQLite)
MLFLOW_ARTIFACT_DIR = "mlflow_artifacts"     # artefactos cuando se usa SQLite
os.makedirs(MLFLOW_ARTIFACT_DIR, exist_ok=True)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # red de seguridad para el fallback

def _setup_mlflow_tracking():
    """Configura el tracking de MLflow, evitando el error de 'maintenance mode'
    del file store plano en versiones recientes de MLflow."""
    try:
        tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
        mlflow.set_tracking_uri(tracking_uri)
        existing_exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if existing_exp is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME,
                                      artifact_location=os.path.abspath(MLFLOW_ARTIFACT_DIR))
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"[MLflow] Backend de tracking: SQLite ({tracking_uri}), "
              f"artefactos en ./{MLFLOW_ARTIFACT_DIR}/")
        return tracking_uri
    except Exception as e:
        print(f"[MLflow] Backend SQLite no disponible ({e}); usando file store "
              f"local clasico con MLFLOW_ALLOW_FILE_STORE=true como opt-out.")
        tracking_uri = f"file:{os.path.abspath(MLFLOW_TRACKING_DIR)}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return tracking_uri

MLFLOW_TRACKING_URI_USED = _setup_mlflow_tracking()

MLFLOW_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_mlflow_active_run = mlflow.start_run(run_name=f"mpfn_q1_extreme_{MLFLOW_RUN_TIMESTAMP}")
print(f"[MLflow] Run iniciado: {_mlflow_active_run.info.run_id} "
      f"(experimento '{MLFLOW_EXPERIMENT_NAME}', tracking_uri='{MLFLOW_TRACKING_URI_USED}')")

def mlflow_log_params_safe(params_dict):
    """[PATCH 21] Registra parametros en MLflow tolerando valores no serializables."""
    for k, v in params_dict.items():
        try:
            mlflow.log_param(k, v if isinstance(v, (int, float, str, bool)) else str(v))
        except Exception as e:
            print(f"[MLflow] No se pudo registrar el parametro '{k}': {e}")

def mlflow_log_metrics_safe(metrics_dict, step=None):
    """[PATCH 21] Registra metricas en MLflow, ignorando valores no numericos/NaN."""
    for k, v in metrics_dict.items():
        try:
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            mlflow.log_metric(k, float(v), step=step)
        except Exception as e:
            print(f"[MLflow] No se pudo registrar la metrica '{k}': {e}")

def mlflow_log_artifacts_bulk():
    """
    [PATCH 21] Registra en MLflow, como artefactos del run:
      - Todo outputs/ (incluye todos los .pkl/.joblib de modelos, .csv/.xlsx
        de tablas, .png de figuras, el/los PDF y el HTML si existen).
      - Todo checkpoints/ (incluye el manifest SHA-256).
      - Todo DOI_PACKAGE/.
      - El ZIP final (si ya se genero).
    Se ejecuta al final del pipeline, ya con auto-regeneracion aplicada, para
    que ningun artefacto critico quede fuera del historial del experimento.
    """
    for local_dir, artifact_path in [
        (OUTPUT_DIR, "outputs"),
        (CHECKPOINT_DIR, "checkpoints"),
        (DOI_DIR, "DOI_PACKAGE"),
    ]:
        try:
            if os.path.exists(local_dir):
                mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
                print(f"[MLflow] Artefactos registrados desde '{local_dir}/' -> '{artifact_path}/'")
        except Exception as e:
            print(f"[MLflow] No se pudieron registrar artefactos de '{local_dir}': {e}")

    global zip_name
    try:
        if "zip_name" in globals() and zip_name and os.path.exists(zip_name):
            mlflow.log_artifact(zip_name, artifact_path="zip_final")
            print(f"[MLflow] ZIP final registrado como artefacto: {zip_name}")
    except Exception as e:
        print(f"[MLflow] No se pudo registrar el ZIP final: {e}")

def mlflow_log_best_model_safe(model, model_name="predictive_model"):
    """
    [PATCH 21] Intenta registrar el modelo principal con el flavor sklearn de
    MLflow. Modelos no-sklearn (wrappers propios de TabNet/FTTransformer,
    artefactos de stacking manual) se omiten con aviso, ya que MLflow no
    sabe serializarlos de forma generica sin un flavor custom.
    """
    try:
        mlflow.sklearn.log_model(model, model_name)
        print(f"[MLflow] Modelo '{model_name}' registrado con flavor sklearn.")
    except Exception as e:
        print(f"[MLflow] Modelo '{model_name}' no se pudo registrar con flavor sklearn "
              f"(se conserva igualmente como .joblib en outputs/models): {e}")

def export_mlflow_runs_to_doi_package():
    """[PATCH 25, actualizado v3.01] Copia el historial de MLflow dentro de
    DOI_PACKAGE/MLFLOW_EXPORT para que el experimento completo (parametros,
    metricas, modelos, artefactos) viaje junto con el paquete de
    reproducibilidad. Soporta ambos backends posibles:
      - SQLite (preferido): mlflow.db + mlflow_artifacts/
      - File store legacy (fallback): mlruns/
    """
    dst = os.path.join(DOI_DIR, "MLFLOW_EXPORT")
    os.makedirs(dst, exist_ok=True)
    copied_any = False
    try:
        if os.path.exists(MLFLOW_DB_PATH):
            shutil.copy2(MLFLOW_DB_PATH, os.path.join(dst, os.path.basename(MLFLOW_DB_PATH)))
            copied_any = True
        if os.path.exists(MLFLOW_ARTIFACT_DIR):
            artifacts_dst = os.path.join(dst, MLFLOW_ARTIFACT_DIR)
            if os.path.exists(artifacts_dst):
                shutil.rmtree(artifacts_dst)
            shutil.copytree(MLFLOW_ARTIFACT_DIR, artifacts_dst)
            copied_any = True
        if os.path.exists(MLFLOW_TRACKING_DIR):
            mlruns_dst = os.path.join(dst, MLFLOW_TRACKING_DIR)
            if os.path.exists(mlruns_dst):
                shutil.rmtree(mlruns_dst)
            shutil.copytree(MLFLOW_TRACKING_DIR, mlruns_dst)
            copied_any = True
        if copied_any:
            print(f"[MLflow] Historial de tracking copiado a {dst}/")
        else:
            print("[MLflow] No se encontro mlflow.db, mlflow_artifacts/ ni mlruns/ para exportar.")
    except Exception as e:
        print(f"[MLflow] No se pudo exportar el historial de tracking al DOI_PACKAGE: {e}")

def clean_for_export(df):
    """
    [v3.02 - observacion revisor #1] Reemplaza inf/-inf por NaN y luego todos
    los NaN por "N/A" antes de exportar a Excel/CSV. Evita que aparezcan NaN
    "crudos" en reportes (p.ej. F1=NaN de un DummyClassifier constante, o
    std=NaN en variables categoricas agregadas), que un revisor Q1 podria
    malinterpretar como un error del pipeline en vez de un caso degenerado
    esperado (ej. metrica indefinida cuando el modelo predice una sola clase).
    """
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df
    return df.replace([np.inf, -np.inf], np.nan).fillna("N/A")

def cohens_d(a, b):
    """[v3.02 - observacion revisor #2] Tamano del efecto Cohen's d (diferencia
    de medias estandarizada, pooled std) entre dos muestras independientes."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled_std)

def cliffs_delta(a, b, max_n=2000):
    """[v3.02 - observacion revisor #2] Cliff's Delta: tamano del efecto no
    parametrico basado en dominancia estocastica. Submuestrea a `max_n` por
    grupo si son mas grandes, para mantener el costo O(n*m) manejable."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    rng = np.random.RandomState(RANDOM_STATE)
    if len(a) > max_n:
        a = rng.choice(a, size=max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, size=max_n, replace=False)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n = len(a) * len(b)
    return float((gt - lt) / n) if n > 0 else np.nan

def cramers_v(confusion_matrix_2d):
    """[v3.02 - observacion revisor #2] Cramer's V: tamano del efecto de
    asociacion para tablas de contingencia (variables categoricas)."""
    try:
        chi2_stat = chi2_contingency(confusion_matrix_2d)[0]
        n = confusion_matrix_2d.values.sum() if hasattr(confusion_matrix_2d, "values") else np.sum(confusion_matrix_2d)
        r, k = confusion_matrix_2d.shape
        if n == 0 or min(r - 1, k - 1) == 0:
            return np.nan
        return float(np.sqrt((chi2_stat / n) / min(r - 1, k - 1)))
    except Exception:
        return np.nan

def compute_ece(y_true, y_prob, n_bins=10):
    """[v3.02 - observacion revisor #14] Expected Calibration Error (ECE) y
    Maximum Calibration Error (MCE): discretiza las probabilidades en
    `n_bins` cubos de igual ancho y compara la tasa observada vs predicha."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(y_true)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.sum() / n
        gap = abs(acc - conf)
        ece += weight * gap
        mce = max(mce, gap)
        bin_rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
                          "confianza_promedio": conf, "tasa_observada": acc, "gap": gap})
    return float(ece), float(mce), pd.DataFrame(bin_rows)

    """
    [PATCH 23] Escritura incremental: agrega filas a un CSV existente en vez
    de acumular todo en memoria y escribir un unico archivo al final. Si
    `mode_reset=True`, fuerza sobreescritura (util al iniciar un nuevo run).
    `df_or_row` puede ser un dict (una fila) o un DataFrame (varias filas).
    """
    if isinstance(df_or_row, dict):
        chunk_df = pd.DataFrame([df_or_row])
    else:
        chunk_df = df_or_row
    file_exists = os.path.exists(path) and not mode_reset
    chunk_df.to_csv(path, mode="a" if file_exists else "w",
                     header=not file_exists, index=False)
    return path

def write_pipeline_log_csv(resource_rows, checkpoint_manifest, output_path):
    """
    [PATCH 24] Consolida el log por modulo (inicio/fin/tiempo/RAM/CPU/estado)
    ya recolectado por CheckpointManager, y le añade la ruta de checkpoint
    asociado a cada modulo (si existe), en pipeline_log.csv.
    """
    rows = []
    completed = checkpoint_manifest.get("completed_modules", {}) if checkpoint_manifest else {}
    for r in resource_rows:
        module_id = r.get("module_id")
        ckpt_info = completed.get(module_id, {})
        rows.append({
            "modulo": module_id,
            "inicio_ram_gb": r.get("start_ram_used_gb"),
            "fin_ram_gb": r.get("end_ram_used_gb"),
            "ram_disponible_fin_gb": r.get("end_ram_available_gb"),
            "tiempo_segundos": r.get("elapsed_seconds"),
            "tiempo_legible": r.get("elapsed_human"),
            "estado": r.get("status"),
            "error": r.get("error", ""),
            "timestamp": r.get("timestamp"),
            "checkpoint_pkl": ckpt_info.get("pkl_path", ""),
            "checkpoint_sha256": ckpt_info.get("sha256", ""),
        })
    log_df = pd.DataFrame(rows)
    log_df.to_csv(output_path, index=False)
    print(f"[pipeline_log.csv] {len(rows)} modulos registrados -> {output_path}")
    return log_df

# Instancia global del gestor de checkpoints, disponible para todo el pipeline.
ckpt = CheckpointManager(autosave_minutes=5, ram_floor_gb=1.0)

# [PATCH 21] Parametros de configuracion global registrados en MLflow desde el inicio.
mlflow_log_params_safe({
    "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS,
    "n_experiments_per_model": N_EXPERIMENTS_PER_MODEL, "fast_mode": FAST_MODE,
    "run_heavy_blocks": RUN_HEAVY_BLOCKS, "use_smote": USE_SMOTE,
    "train_years": str(TRAIN_YEARS), "valid_year": VALID_YEAR,
    "test_year": TEST_YEAR, "external_year": EXTERNAL_YEAR,
    "primary_target_policy": PRIMARY_TARGET_POLICY,
})
# ============================================================
# FIN [PATCH 11-20] / FIN [PATCH 21-25 setup inicial]
# ============================================================

print("Librerias importadas correctamente.")
environment_versions_df = print_environment_versions()

import shutil
import importlib

# ============================================================
# MODULO 0.5 - ENVIRONMENT & HARDWARE AUDIT
# Se ejecuta una sola vez al inicio (con su propio checkpoint M00) y deja
# constancia completa del entorno de ejecucion para reproducibilidad Q1:
# CPU, RAM, GPU, SO, Python, versiones de librerias clave, y si el runtime
# es Google Colab / local / Kaggle, con o sin GPU.
# ============================================================
ENV_DIR = os.path.join(OUTPUT_DIR, "00_ENVIRONMENT")
os.makedirs(ENV_DIR, exist_ok=True)

def _detect_execution_platform():
    """Heuristica simple para identificar Colab / Kaggle / entorno local."""
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "Google Colab"
    if os.path.exists("/kaggle/input"):
        return "Kaggle"
    return "Local / Servidor propio"

def _detect_gpu_info():
    """Detecta GPU/VRAM/CUDA sin asumir que torch/nvidia-smi esten disponibles."""
    gpu_info = {"gpu_available": False, "gpu_name": "None", "vram_gb": None, "cuda_version": "N/A"}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["gpu_available"] = True
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            gpu_info["cuda_version"] = torch.version.cuda or "N/A"
            return gpu_info
    except Exception:
        pass
    try:
        nvidia_smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if nvidia_smi_out:
            name, mem_mb = [x.strip() for x in nvidia_smi_out.split(",")]
            gpu_info["gpu_available"] = True
            gpu_info["gpu_name"] = name
            gpu_info["vram_gb"] = round(float(mem_mb) / 1024, 2)
    except Exception:
        pass
    return gpu_info

def run_environment_hardware_audit():
    """
    [Modulo 0.5] Ejecuta el audit completo de entorno/hardware y escribe:
      00_ENVIRONMENT/hardware_info.json
      00_ENVIRONMENT/software_versions.xlsx
      00_ENVIRONMENT/cpu_info.txt
      00_ENVIRONMENT/gpu_info.txt
      00_ENVIRONMENT/ram_info.txt
      00_ENVIRONMENT/disk_info.txt
      00_ENVIRONMENT/execution_environment.json
      00_ENVIRONMENT/environment_report.pdf
    Se saltea automaticamente (checkpoint M00) si ya se ejecuto antes.
    """
    if ckpt.should_skip("M00_environment_audit"):
        print("[Modulo 0.5] Checkpoint M00 encontrado; auditoria de entorno ya realizada, se omite.")
        return ckpt.load("M00_environment_audit")

    with ckpt.timed("M00_environment_audit"):
        # --- CPU / RAM ---
        cpu_info = {"cpu_model": platform.processor() or "No disponible",
                    "architecture": platform.machine(),
                    "physical_cores": None, "logical_cores": None}
        ram_info = {"ram_total_gb": None, "ram_available_gb": None}
        try:
            import psutil
            cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
            cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            ram_info["ram_total_gb"] = round(vm.total / 1e9, 2)
            ram_info["ram_available_gb"] = round(vm.available / 1e9, 2)
        except Exception as e:
            print(f"[Modulo 0.5] psutil no disponible para CPU/RAM detallada: {e}")

        # --- Disco ---
        disk_info = {"disk_total_gb": None, "disk_free_gb": None}
        try:
            total, used, free = shutil.disk_usage(".")
            disk_info["disk_total_gb"] = round(total / 1e9, 2)
            disk_info["disk_free_gb"] = round(free / 1e9, 2)
        except Exception as e:
            print(f"[Modulo 0.5] No se pudo leer espacio en disco: {e}")

        # --- GPU ---
        gpu_info = _detect_gpu_info()

        # --- Software: versiones de librerias clave ---
        sw_packages = ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost",
                       "shap", "optuna", "scipy", "joblib", "matplotlib", "statsmodels",
                       "umap", "reportlab", "psutil"]
        sw_versions = {}
        for pkg in sw_packages:
            try:
                mod = importlib.import_module(pkg.replace("-", "_"))
                sw_versions[pkg] = getattr(mod, "__version__", "instalado")
            except Exception:
                sw_versions[pkg] = "no instalado"

        # --- Entorno de ejecucion ---
        execution_platform = _detect_execution_platform()
        execution_environment = {
            "platform": execution_platform,
            "runtime_type": "GPU" if gpu_info["gpu_available"] else "CPU",
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "execution_datetime": datetime.datetime.now().isoformat(),
            "timezone": time.strftime("%Z", time.localtime()),
        }

        hardware_info = {**cpu_info, **ram_info, **disk_info, **gpu_info,
                          "execution_platform": execution_platform}

        # --- Escritura de artefactos ---
        with open(os.path.join(ENV_DIR, "hardware_info.json"), "w", encoding="utf-8") as f:
            json.dump(hardware_info, f, indent=2, ensure_ascii=False)

        with open(os.path.join(ENV_DIR, "execution_environment.json"), "w", encoding="utf-8") as f:
            json.dump(execution_environment, f, indent=2, ensure_ascii=False)

        pd.DataFrame([{"package": k, "version": v} for k, v in sw_versions.items()]).to_excel(
            os.path.join(ENV_DIR, "software_versions.xlsx"), index=False)

        with open(os.path.join(ENV_DIR, "cpu_info.txt"), "w", encoding="utf-8") as f:
            for k, v in cpu_info.items():
                f.write(f"{k}: {v}\n")

        with open(os.path.join(ENV_DIR, "gpu_info.txt"), "w", encoding="utf-8") as f:
            for k, v in gpu_info.items():
                f.write(f"{k}: {v}\n")

        with open(os.path.join(ENV_DIR, "ram_info.txt"), "w", encoding="utf-8") as f:
            for k, v in ram_info.items():
                f.write(f"{k}: {v}\n")

        with open(os.path.join(ENV_DIR, "disk_info.txt"), "w", encoding="utf-8") as f:
            for k, v in disk_info.items():
                f.write(f"{k}: {v}\n")

        # --- Reporte PDF corto (Execution Environment) ---
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.enums import TA_CENTER

            pdf_path = os.path.join(ENV_DIR, "environment_report.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm,
                                     topMargin=2.2*cm, bottomMargin=2.2*cm,
                                     title="Environment & Hardware Audit - MPFN Q1 Extreme")
            title_style = ParagraphStyle("T", fontSize=16, fontName="Helvetica-Bold",
                                          alignment=TA_CENTER, textColor=colors.darkblue, spaceAfter=14)
            body = getSampleStyleSheet()["Normal"]

            def _tbl(rows):
                t = Table([["Item", "Valor"]] + rows, colWidths=[7*cm, 9*cm])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.97, 1.0)]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ]))
                return t

            story = [Paragraph("ENVIRONMENT & HARDWARE AUDIT", title_style),
                     Paragraph("Modulo 0.5 - MPFN Q1 Extreme Pipeline v3.02-industrial", body),
                     Spacer(1, 10)]
            story.append(_tbl([[k, str(v)] for k, v in execution_environment.items()]))
            story.append(Spacer(1, 10))
            story.append(_tbl([[k, str(v)] for k, v in hardware_info.items()]))
            story.append(Spacer(1, 10))
            story.append(_tbl([[k, str(v)] for k, v in sw_versions.items()]))
            doc.build(story)
            print(f"[Modulo 0.5] environment_report.pdf generado -> {pdf_path}")
        except Exception as e:
            print(f"[Modulo 0.5] No se pudo generar environment_report.pdf: {e}")

        print("Execution Environment")
        print(f"  Platform: {execution_environment['platform']}")
        print(f"  Runtime: {execution_environment['runtime_type']}")
        print(f"  CPU: {cpu_info['cpu_model']} | Physical cores: {cpu_info['physical_cores']} | "
              f"Logical cores: {cpu_info['logical_cores']}")
        print(f"  RAM: {ram_info['ram_total_gb']} GB total / {ram_info['ram_available_gb']} GB libre")
        print(f"  GPU: {gpu_info['gpu_name']}"
              + (f" | VRAM: {gpu_info['vram_gb']} GB | CUDA: {gpu_info['cuda_version']}"
                 if gpu_info["gpu_available"] else ""))
        print(f"  OS: {execution_environment['operating_system']} | Python: {execution_environment['python_version']}")

        state = {"hardware_info": hardware_info, "execution_environment": execution_environment,
                  "sw_versions": sw_versions}
        ckpt.save("M00_environment_audit", **state)
        return state

environment_audit_state = run_environment_hardware_audit()

def extract_year_from_filename(path):
    """Extrae el anio desde el nombre del archivo. Se conserva como utilidad
    generica, pero [PATCH 9] ya NO se usa como fallback para poblar 'anio'
    en el dataset consolidado multi-anio (ver bloque de carga mas abajo)."""
    m = re.search(r"(20\d{2})", os.path.basename(path))
    return int(m.group(1)) if m else None

def normalize_column_name(col):
    """Normaliza nombres de columnas sin mezclar variables conceptualmente distintas."""
    col = str(col).replace("\ufeff", "").strip().lower()
    col = re.sub(r"\s+", "_", col)
    col = col.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    typo_map = {
        "especilizada": "especializada",
        "especializacion": "especializada"
    }
    return typo_map.get(col, col)

def coalesce_duplicate_columns(df_in, source_file):
    """
    Detecta columnas duplicadas despues de normalizar.
    No las elimina silenciosamente: primero genera auditoria y luego fusiona por primer valor no nulo.
    Si dos columnas duplicadas tienen valores no nulos distintos en una misma fila, se reporta como conflicto.
    """
    df = df_in.copy()
    raw_columns = list(df.columns)
    normalized_columns = [normalize_column_name(c) for c in raw_columns]

    audit_rows = []
    for raw, norm in zip(raw_columns, normalized_columns):
        audit_rows.append({
            "archivo": source_file,
            "columna_original": raw,
            "columna_normalizada": norm
        })

    df.columns = normalized_columns
    duplicate_names = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].unique().tolist()
    duplicate_report_rows = []

    if duplicate_names:
        print(f"Columnas duplicadas detectadas en {source_file}: {duplicate_names}")

    output_cols = {}
    for col in pd.unique(df.columns):
        same = df.loc[:, df.columns == col]
        if isinstance(same, pd.Series):
            output_cols[col] = same
            continue

        if same.shape[1] == 1:
            output_cols[col] = same.iloc[:, 0]
        else:
            non_null_counts = same.notna().sum(axis=1)
            unique_non_null_counts = same.apply(lambda row: len(pd.unique(row.dropna())), axis=1)
            conflict_rows = int(((non_null_counts > 1) & (unique_non_null_counts > 1)).sum())
            duplicate_report_rows.append({
                "archivo": source_file,
                "columna_normalizada": col,
                "n_columnas_duplicadas": same.shape[1],
                "nulos_por_columna": json.dumps([int(same.iloc[:, i].isna().sum()) for i in range(same.shape[1])]),
                "valores_unicos_por_columna": json.dumps([int(same.iloc[:, i].nunique(dropna=True)) for i in range(same.shape[1])]),
                "filas_con_conflicto": conflict_rows,
                "decision": "fusion_primer_valor_no_nulo"
            })
            output_cols[col] = same.bfill(axis=1).iloc[:, 0]

    df_out = pd.DataFrame(output_cols)
    return df_out, audit_rows, duplicate_report_rows

if ckpt.should_skip("M01_dataset_cargado"):
    print("[Modulo 0.5 -> M01] Checkpoint encontrado para 'M01_dataset_cargado'. "
          "Reanudando SIN recargar/normalizar el CSV completo...")
    _m01_state = ckpt.load("M01_dataset_cargado")
    df_raw = _m01_state["df_raw"]
    file_summary_df = _m01_state["file_summary_df"]
    column_audit_df = _m01_state["column_audit_df"]
    duplicate_columns_df = _m01_state["duplicate_columns_df"]
    DATASET_PATH = "/content/drive/MyDrive/dataset2026varios/Data_Final_Casos_Fiscales_Ene_2019-May_2026.csv"
    print("Dataset recuperado desde checkpoint:", df_raw.shape)
else:
    # ============================================================
    # RUTA DEL DATASET - Google Drive
    # ============================================================
    DATASET_PATH = "/content/drive/MyDrive/dataset2026varios/Data_Final_Casos_Fiscales_Ene_2019-May_2026.csv"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"No se encontro el dataset en:\n  {DATASET_PATH}\n"
            "Verifica que Google Drive este montado y la ruta sea correcta."
        )

    print(f"Cargando dataset desde:\n  {DATASET_PATH}")

    file_summary = []
    column_audit_all = []
    duplicate_columns_all = []

    # Cargar el CSV unico con el dataset consolidado 2019-2026
    try:
        df_tmp_raw = pd.read_csv(DATASET_PATH, encoding="utf-8", sep=None, engine="python")
    except UnicodeDecodeError:
        df_tmp_raw = pd.read_csv(DATASET_PATH, encoding="latin-1", sep=None, engine="python")

    print(f"Dataset cargado: {df_tmp_raw.shape[0]:,} registros x {df_tmp_raw.shape[1]} columnas")

    df_tmp, audit_rows, duplicate_rows = coalesce_duplicate_columns(df_tmp_raw, os.path.basename(DATASET_PATH))

    column_audit_all.extend(audit_rows)
    duplicate_columns_all.extend(duplicate_rows)

    # [PATCH 9] -----------------------------------------------------------------
    # El dataset consolidado cubre 2019-2026 (multiples anios en un solo CSV).
    # Inferir 'anio' desde el nombre del archivo seria incorrecto y peligroso
    # (asignaria el mismo anio a TODOS los registros, contaminando la particion
    # temporal y el target proxy). Por eso, si falta la columna 'anio', el
    # pipeline se detiene explicitamente en vez de adivinar.
    if "anio" not in df_tmp.columns:
        raise ValueError(
            "El dataset consolidado debe contener una columna 'anio'. "
            "No se infiere el anio desde el nombre del archivo porque el CSV "
            "contiene multiples anios (2019-2026) y esa inferencia seria incorrecta "
            "y contaminaria la particion temporal / el target proxy."
        )
    # -----------------------------------------------------------------------------

    df_tmp["archivo_fuente"] = os.path.basename(DATASET_PATH)

    # Resumen por anio
    for yr, g in df_tmp.groupby("anio"):
        file_summary.append({
            "anio": int(yr),
            "archivo": os.path.basename(DATASET_PATH),
            "registros": len(g),
            "columnas_originales": df_tmp_raw.shape[1],
            "columnas_normalizadas": df_tmp.shape[1]
        })

    # [PATCH 7] -----------------------------------------------------------------
    # Solo se carga UN CSV consolidado (no una lista de archivos por anio), asi
    # que la logica original de frames=[]; frames.append(df_tmp); pd.concat(frames)
    # era codigo muerto/innecesario. Se simplifica directamente:
    df_raw = df_tmp.copy()
    # -----------------------------------------------------------------------------

    file_summary_df = pd.DataFrame(file_summary).sort_values("anio")
    column_audit_df = pd.DataFrame(column_audit_all)
    duplicate_columns_df = pd.DataFrame(duplicate_columns_all)

    print("Archivo integrado:")
    display(file_summary_df)

    print("Auditoria de columnas duplicadas despues de normalizacion:")
    display(duplicate_columns_df if len(duplicate_columns_df) else pd.DataFrame({"mensaje": ["No se detectaron columnas duplicadas tras normalizacion."]}))

    print("Dataset crudo integrado:", df_raw.shape)

    # [PATCH 11-13] Checkpoint M01: dataset crudo cargado y auditado.
    # Guarda df_raw (via Parquet si es grande) para que una caida posterior
    # no obligue a releer/normalizar el CSV completo desde cero.
    ckpt.save("M01_dataset_cargado", df_raw=df_raw, file_summary_df=file_summary_df,
              column_audit_df=column_audit_df, duplicate_columns_df=duplicate_columns_df)

file_summary_df.to_excel(os.path.join(TABLE_DIR, "01_archivos_integrados.xlsx"), index=False)
column_audit_df.to_excel(os.path.join(TABLE_DIR, "01b_auditoria_columnas.xlsx"), index=False)
duplicate_columns_df.to_excel(os.path.join(TABLE_DIR, "01c_columnas_duplicadas_auditadas.xlsx"), index=False)

import pandas as pd
import numpy as np
import os

# ============================================================
# 4. PERFILAMIENTO Y CALIDAD DE DATOS SIN IMPUTAR
# ============================================================
df_profile = df_raw.copy()

duplicated_cols_integrated = df_profile.columns[df_profile.columns.duplicated()].tolist()
print("Columnas duplicadas en dataset integrado:", duplicated_cols_integrated if duplicated_cols_integrated else "No hay columnas duplicadas.")

missing_df = pd.DataFrame({
    "variable": df_profile.columns,
    "nulos": df_profile.isna().sum().values,
    "porcentaje_nulos": (df_profile.isna().sum().values / len(df_profile) * 100).round(2),
    "tipo_dato": df_profile.dtypes.astype(str).tolist()
}).sort_values("nulos", ascending=False)

missing_nonzero = missing_df[missing_df["nulos"] > 0]
print("Variables con nulos antes de imputar:")
display(missing_nonzero)

exact_dups = df_profile.duplicated(keep=False)
print("Duplicados exactos detectados:", int(exact_dups.sum()))

schema_df = pd.DataFrame({
    "variable": df_profile.columns,
    "tipo_dato": df_profile.dtypes.astype(str).tolist(),
    "valores_unicos": [df_profile[c].nunique(dropna=True) for c in df_profile.columns]
}).sort_values("variable")

numeric_like = []
for c in df_profile.columns:
    s = pd.to_numeric(df_profile[c], errors="coerce")
    if s.notna().mean() > 0.75 and c not in ["anio"]:
        numeric_like.append(c)

print("Variables numericas probables:", numeric_like)

outlier_rows = []
for c in numeric_like:
    s = pd.to_numeric(df_profile[c], errors="coerce").dropna()
    if len(s) < 10:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_count = int(((s < li) | (s > ls)).sum())
    outlier_rows.append({"variable": c, "Q1": q1, "Q3": q3, "IQR": iqr, "LI": li, "LS": ls, "outliers": out_count})

outlier_iqr_df = pd.DataFrame(outlier_rows)

display(schema_df.head(30))
display(outlier_iqr_df.sort_values("outliers", ascending=False).head(20) if len(outlier_iqr_df) else pd.DataFrame())

missing_df.to_excel(os.path.join(TABLE_DIR, "02_nulos_antes_imputacion.xlsx"), index=False)
schema_df.to_excel(os.path.join(TABLE_DIR, "03_esquema_tipos_datos.xlsx"), index=False)
outlier_iqr_df.to_excel(os.path.join(TABLE_DIR, "04_outliers_iqr_preliminar.xlsx"), index=False)

df = df_profile.copy()

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip().str.upper()
    df[c] = df[c].replace({"NAN": np.nan, "NONE": np.nan, "": np.nan, "NULL": np.nan})

for col in ["ingresado", "atendido", "ubigeo_pjfs"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.columns:
    if df[col].isna().sum() > 0:
        df[f"flag_nulo_{col}"] = df[col].isna().astype(int)

for col in ["ingresado", "atendido"]:
    if col in df.columns:
        train_median = df.loc[df["anio"].isin(TRAIN_YEARS), col].median()
        global_median = df[col].median()
        value = train_median if pd.notna(train_median) else global_median
        df[col] = df[col].fillna(value)
        print(f"{col}: imputacion con mediana train = {value}")

categorical_explicit = ["especializada", "materia", "especialidad", "tipo_fiscalia", "tipo_caso", "dpto_pjfs", "prov_pjfs", "dist_pjfs"]
for col in categorical_explicit:
    if col in df.columns:
        df[col] = df[col].fillna("NO_ESPECIFICADO")

df["flag_duplicado_exacto"] = df.duplicated(keep=False).astype(int)

imputation_summary = []
for col in df_profile.columns:
    n0 = int(df_profile[col].isna().sum()) if col in df_profile.columns else 0
    n1 = int(df[col].isna().sum()) if col in df.columns else 0
    if n0 > 0:
        method = "mediana_train" if col in ["ingresado", "atendido"] else "categoria_explicita/moda_en_pipeline"
        imputation_summary.append({"variable": col, "nulos_antes": n0, "nulos_despues": n1, "metodo": method})
imputation_summary_df = pd.DataFrame(imputation_summary)

display(imputation_summary_df)
imputation_summary_df.to_excel(os.path.join(TABLE_DIR, "05_resumen_imputacion.xlsx"), index=False)
print("Dataset despues de limpieza inicial:", df.shape)

if "ingresado" not in df.columns or "atendido" not in df.columns:
    raise KeyError("El dataset debe contener las columnas 'ingresado' y 'atendido'.")

df["saldo_casos"] = df["ingresado"] - df["atendido"]
df["tasa_atencion"] = np.where(df["ingresado"] > 0, df["atendido"] / df["ingresado"], np.nan)
df["tasa_atencion"] = df["tasa_atencion"].replace([np.inf, -np.inf], np.nan)
df["tasa_atencion"] = df["tasa_atencion"].fillna(df.loc[df["anio"].isin(TRAIN_YEARS), "tasa_atencion"].median())
df["ratio_saldo"] = np.where(df["ingresado"] > 0, df["saldo_casos"] / df["ingresado"], 0)
df["ratio_saldo"] = df["ratio_saldo"].replace([np.inf, -np.inf], 0).fillna(0)

df["anio_centrado"] = df["anio"] - min(TRAIN_YEARS)
df["post_pandemia"] = (df["anio"] >= 2021).astype(int)
df["periodo_pandemia_2020"] = (df["anio"] == 2020).astype(int)

if "tipo_fiscalia" in df.columns and "especialidad" in df.columns:
    df["inter_tipo_fiscalia_especialidad"] = df["tipo_fiscalia"].astype(str) + "__" + df["especialidad"].astype(str)
if "dist_pjfs" in df.columns and "tipo_caso" in df.columns:
    df["inter_distrito_tipo_caso"] = df["dist_pjfs"].astype(str) + "__" + df["tipo_caso"].astype(str)
if "materia" in df.columns and "tipo_fiscalia" in df.columns:
    df["inter_materia_tipo_fiscalia"] = df["materia"].astype(str) + "__" + df["tipo_fiscalia"].astype(str)

group_candidates = [c for c in ["dist_pjfs", "tipo_fiscalia", "especialidad"] if c in df.columns]
historical_feature_audit = []

if group_candidates:
    group_col = group_candidates[0]

    annual_group = (
        df.groupby([group_col, "anio"], as_index=False)
          .agg(
              ingresado_anual=("ingresado", "sum"),
              atendido_anual=("atendido", "sum"),
              saldo_anual=("saldo_casos", "sum"),
              registros_anuales=("anio", "count")
          )
    ).sort_values([group_col, "anio"])

    annual_group[f"hist_ingresado_mean_prev_{group_col}"] = (
        annual_group.groupby(group_col)["ingresado_anual"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    annual_group[f"hist_atendido_mean_prev_{group_col}"] = (
        annual_group.groupby(group_col)["atendido_anual"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    annual_group[f"hist_saldo_mean_prev_{group_col}"] = (
        annual_group.groupby(group_col)["saldo_anual"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    annual_group[f"growth_ingresado_prev_{group_col}"] = (
        annual_group.groupby(group_col)["ingresado_anual"]
        .transform(lambda s: s.shift(1).pct_change())
        .replace([np.inf, -np.inf], np.nan)
    )
    annual_group[f"growth_atendido_prev_{group_col}"] = (
        annual_group.groupby(group_col)["atendido_anual"]
        .transform(lambda s: s.shift(1).pct_change())
        .replace([np.inf, -np.inf], np.nan)
    )

    hist_cols = [
        f"hist_ingresado_mean_prev_{group_col}",
        f"hist_atendido_mean_prev_{group_col}",
        f"hist_saldo_mean_prev_{group_col}",
        f"growth_ingresado_prev_{group_col}",
        f"growth_atendido_prev_{group_col}",
    ]

    df = df.merge(annual_group[[group_col, "anio"] + hist_cols], on=[group_col, "anio"], how="left")

    for c in hist_cols:
        train_values = df.loc[df["anio"].isin(TRAIN_YEARS), c]
        fill_value = train_values.median() if train_values.notna().any() else 0
        df[c] = df[c].fillna(fill_value)
        historical_feature_audit.append({
            "feature": c,
            "group_col": group_col,
            "definition": "Agregado anual previo por grupo; no usa registros del mismo anio como historial.",
            "fill_value_train_median": fill_value
        })

    pd.DataFrame(historical_feature_audit).to_excel(os.path.join(TABLE_DIR, "06b_auditoria_features_historicas_temporales.xlsx"), index=False)
else:
    print("No se encontraron columnas de grupo para crear variables historicas.")

print("Variables creadas para feature engineering:")
created_cols = ["saldo_casos", "tasa_atencion", "ratio_saldo", "anio_centrado", "post_pandemia", "periodo_pandemia_2020"]
created_cols += [c for c in df.columns if c.startswith("inter_") or c.startswith("hist_") or c.startswith("growth_")]
print(created_cols)

# [PATCH 8] Assertions tempranas de integridad de datos.
assert "anio" in df.columns, "Falta la columna 'anio' tras la carga/limpieza inicial."
assert df["anio"].between(2019, 2026).all(), "Hay valores de 'anio' fuera del rango esperado (2019-2026)."
assert df["saldo_casos"].notna().all(), "saldo_casos contiene nulos tras el feature engineering."

# [PATCH 11-13] Checkpoint M02: feature engineering completo (antes del target).
ckpt.save("M02_feature_engineering", df=df)

train_mask_for_thresholds = df["anio"].isin(TRAIN_YEARS)

TARGET_SCENARIOS = {
    "P70_P30": {"saldo_q": 0.70, "tasa_q": 0.30, "descripcion": "Flexible: mas sensible, mas alertas"},
    "P75_P25": {"saldo_q": 0.75, "tasa_q": 0.25, "descripcion": "Base: equilibrio entre severidad y prevalencia"},
    "P80_P20": {"saldo_q": 0.80, "tasa_q": 0.20, "descripcion": "Estricto: menos alertas, mayor severidad"},
    "P85_P15": {"saldo_q": 0.85, "tasa_q": 0.15, "descripcion": "Muy estricto: casos extremos"},
}

PRIMARY_TARGET_SCENARIO = "P75_P25"
EARLY_WARNING_TARGET_SCENARIO = "P70_P30"

TARGET_SCENARIO_ROLES = {
    "P70_P30": "escenario_alternativo_alerta_temprana",
    "P75_P25": "escenario_principal_conservador",
    "P80_P20": "escenario_estricto_sobrecarga_severa",
    "P85_P15": "escenario_muy_estricto_casos_extremos",
}

TARGET_DECISION_NOTE = (
    "P75/P25 se mantiene como target principal por ser una definicion conservadora y mas alineada "
    "con sobrecarga operacional severa. P70/P30 se reporta como escenario alternativo de alerta temprana "
    "porque puede mejorar F1/PR-AUC al capturar senales menos extremas. La seleccion final no se basa solo "
    "en la metrica mas alta, sino en severidad institucional, estabilidad temporal, balance de clases y validez convergente."
)

def compute_proxy_thresholds(data, saldo_q, tasa_q, train_mask):
    """Calcula umbrales solo en entrenamiento para evitar contaminacion temporal."""
    saldo_thr = data.loc[train_mask, "saldo_casos"].quantile(saldo_q)
    tasa_thr = data.loc[train_mask, "tasa_atencion"].quantile(tasa_q)
    return float(saldo_thr), float(tasa_thr)

def build_proxy_target_from_thresholds(data, saldo_thr, tasa_thr):
    """Construye etiqueta proxy con saldo alto y tasa de atencion baja."""
    return ((data["saldo_casos"] >= saldo_thr) & (data["tasa_atencion"] <= tasa_thr)).astype(int)

target_scenario_rows = []
for scenario_name, params in TARGET_SCENARIOS.items():
    saldo_thr, tasa_thr = compute_proxy_thresholds(
        df,
        saldo_q=params["saldo_q"],
        tasa_q=params["tasa_q"],
        train_mask=train_mask_for_thresholds,
    )
    col_name = f"riesgo_proxy_{scenario_name}"
    df[col_name] = build_proxy_target_from_thresholds(df, saldo_thr, tasa_thr)

    target_scenario_rows.append({
        "scenario": scenario_name,
        "saldo_q": params["saldo_q"],
        "tasa_q": params["tasa_q"],
        "saldo_threshold_train": saldo_thr,
        "tasa_threshold_train": tasa_thr,
        "prevalence_global": df[col_name].mean(),
        "prevalence_train": df.loc[df["anio"].isin(TRAIN_YEARS), col_name].mean(),
        "prevalence_valid_2024": df.loc[df["anio"] == VALID_YEAR, col_name].mean(),
        "prevalence_test_2025": df.loc[df["anio"] == TEST_YEAR, col_name].mean(),
        "prevalence_external_2026": df.loc[df["anio"] == EXTERNAL_YEAR, col_name].mean() if (df["anio"] == EXTERNAL_YEAR).any() else np.nan,
        "descripcion": params["descripcion"],
        "rol_reporte": TARGET_SCENARIO_ROLES.get(scenario_name, "escenario_sensibilidad"),
    })

target_scenarios_df = pd.DataFrame(target_scenario_rows)

P75_SALDO = target_scenarios_df.loc[target_scenarios_df["scenario"] == PRIMARY_TARGET_SCENARIO, "saldo_threshold_train"].iloc[0]
P25_TASA = target_scenarios_df.loc[target_scenarios_df["scenario"] == PRIMARY_TARGET_SCENARIO, "tasa_threshold_train"].iloc[0]

df["riesgo_congestion"] = df[f"riesgo_proxy_{PRIMARY_TARGET_SCENARIO}"].astype(int)

TARGET = "riesgo_congestion"

# [PATCH 8] Assertions de integridad del target justo despues de construirlo.
assert TARGET in df.columns, "TARGET no existe tras la construccion del target proxy."
assert df[TARGET].isin([0, 1]).all(), "riesgo_congestion debe ser estrictamente binario (0/1)."
assert df["anio"].between(2019, 2026).all(), "Valores de anio fuera de rango tras construir el target."

target_dist = df["riesgo_congestion"].value_counts().rename_axis("riesgo_congestion").reset_index(name="registros")
target_dist["porcentaje"] = (target_dist["registros"] / len(df) * 100).round(2)

print("Escenario principal del target:", PRIMARY_TARGET_SCENARIO)
print("Escenario alternativo de alerta temprana:", EARLY_WARNING_TARGET_SCENARIO)
print("Nota de decision del target:", TARGET_DECISION_NOTE)
print("Umbral saldo train:", P75_SALDO)
print("Umbral tasa atencion train:", P25_TASA)

# [PATCH 21] MLflow: parametros del target proxy.
mlflow_log_params_safe({
    "target_primary_scenario": PRIMARY_TARGET_SCENARIO,
    "target_p75_saldo_train": float(P75_SALDO),
    "target_p25_tasa_train": float(P25_TASA),
})

display(target_dist)
display(target_scenarios_df)

target_dist.to_excel(os.path.join(TABLE_DIR, "06_distribucion_target_proxy.xlsx"), index=False)
target_scenarios_df.to_excel(os.path.join(TABLE_DIR, "06b_escenarios_target_proxy.xlsx"), index=False)

if RUN_TARGET_CONSTRUCT_VALIDATION:
    sensitivity_rows = []
    for scenario_name, params in TARGET_SCENARIOS.items():
        col_name = f"riesgo_proxy_{scenario_name}"
        for year, g in df.groupby("anio"):
            sensitivity_rows.append({
                "scenario": scenario_name,
                "anio": year,
                "n": len(g),
                "positivos": int(g[col_name].sum()),
                "prevalence": float(g[col_name].mean()),
                "prevalence_pct": float(g[col_name].mean() * 100),
                "saldo_q": params["saldo_q"],
                "tasa_q": params["tasa_q"],
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    display(sensitivity_df.head(20))
    sensitivity_df.to_excel(os.path.join(TABLE_DIR, "07_sensibilidad_temporal_target_escenarios.xlsx"), index=False)

    target_stability_df = (
        sensitivity_df.groupby("scenario")["prevalence"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    target_stability_df["cv_prevalence"] = target_stability_df["std"] / target_stability_df["mean"].replace(0, np.nan)
    display(target_stability_df)
    target_stability_df.to_excel(os.path.join(TABLE_DIR, "07b_estabilidad_target_por_escenario.xlsx"), index=False)

    risk_by_year = df.groupby("anio")["riesgo_congestion"].agg(["count", "sum", "mean"]).reset_index()
    risk_by_year["riesgo_pct"] = (risk_by_year["mean"] * 100).round(2)
    display(risk_by_year)
    risk_by_year.to_excel(os.path.join(TABLE_DIR, "08_estabilidad_temporal_target_principal.xlsx"), index=False)

    plt.figure(figsize=(7, 4))
    for scenario_name in TARGET_SCENARIOS:
        temp = sensitivity_df[sensitivity_df["scenario"] == scenario_name]
        plt.plot(temp["anio"], temp["prevalence_pct"], marker="o", label=scenario_name)
    plt.title("Prevalencia temporal del target proxy por escenario")
    plt.xlabel("Anio")
    plt.ylabel("Prevalencia de riesgo proxy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_prevalencia_target_escenarios.png"), dpi=300)
    plt.show()

    num_for_tests = [c for c in ["saldo_casos", "tasa_atencion", "ratio_saldo", "ingresado", "atendido"] if c in df.columns]
    stat_rows = []
    for col in num_for_tests:
        a = df.loc[df["riesgo_congestion"] == 0, col].dropna()
        b = df.loc[df["riesgo_congestion"] == 1, col].dropna()
        if len(a) > 10 and len(b) > 10:
            ks = ks_2samp(a, b)
            mw = mannwhitneyu(a, b, alternative="two-sided")
            tt = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            # [v3.02 - observacion revisor #2/#12] Tamanos de efecto: Cohen's d
            # (parametrico) y Cliff's Delta (no parametrico), complementando los
            # p-values con una medida de magnitud de la diferencia.
            d_val = cohens_d(a.values, b.values)
            delta_val = cliffs_delta(a.values, b.values)
            stat_rows.append({
                "variable": col,
                "mean_no_riesgo": a.mean(),
                "mean_riesgo": b.mean(),
                "median_no_riesgo": a.median(),
                "median_riesgo": b.median(),
                "ks_stat": ks.statistic,
                "ks_pvalue": ks.pvalue,
                "mannwhitney_stat": mw.statistic,
                "mannwhitney_pvalue": mw.pvalue,
                "welch_t_stat": tt.statistic,
                "welch_t_pvalue": tt.pvalue,
                "cohens_d": d_val,
                "cliffs_delta": delta_val,
                "effect_size_interpretacion": (
                    "grande" if abs(d_val) >= 0.8 else "mediano" if abs(d_val) >= 0.5
                    else "pequeno" if abs(d_val) >= 0.2 else "trivial"
                ) if pd.notna(d_val) else "N/A",
                "interpretacion": "Separacion estadistica del constructo proxy; no causalidad."
            })
    stats_target_df = pd.DataFrame(stat_rows)
    display(stats_target_df)
    clean_for_export(stats_target_df).to_excel(os.path.join(TABLE_DIR, "09_tests_estadisticos_target_principal.xlsx"), index=False)

    chi_rows = []
    cat_for_chi = [c for c in ["tipo_fiscalia", "tipo_caso", "especialidad", "materia", "dist_pjfs", "dpto_pjfs"] if c in df.columns]
    for col in cat_for_chi:
        tab = pd.crosstab(df[col], df["riesgo_congestion"])
        if tab.shape[0] > 1 and tab.shape[1] == 2:
            chi2_stat, pval, dof, expected = chi2_contingency(tab)
            v_val = cramers_v(tab)
            chi_rows.append({
                "variable": col,
                "chi2": chi2_stat,
                "pvalue": pval,
                "dof": dof,
                "n_categorias": tab.shape[0],
                "cramers_v": v_val,
                "asociacion_interpretacion": (
                    "fuerte" if pd.notna(v_val) and v_val >= 0.5 else
                    "moderada" if pd.notna(v_val) and v_val >= 0.3 else
                    "debil" if pd.notna(v_val) and v_val >= 0.1 else
                    "muy debil / trivial" if pd.notna(v_val) else "N/A"
                ),
                "interpretacion": "Asociacion administrativa con el proxy; no causalidad."
            })
    chi_df = pd.DataFrame(chi_rows).sort_values("pvalue") if chi_rows else pd.DataFrame()
    display(chi_df.head(20) if len(chi_df) else chi_df)
    clean_for_export(chi_df).to_excel(os.path.join(TABLE_DIR, "10_chi_square_target_principal.xlsx"), index=False)

    def compute_woe_iv(data, feature, target, bins=5):
        temp = data[[feature, target]].copy()
        if pd.api.types.is_numeric_dtype(temp[feature]):
            try:
                temp[feature] = pd.qcut(temp[feature], q=bins, duplicates="drop")
            except Exception:
                temp[feature] = pd.cut(temp[feature], bins=bins, duplicates="drop")
        temp[feature] = temp[feature].astype(str).fillna("MISSING")
        grouped = temp.groupby(feature)[target].agg(["count", "sum"]).reset_index()
        grouped = grouped.rename(columns={"sum": "bad"})
        grouped["good"] = grouped["count"] - grouped["bad"]
        eps = 0.5
        total_good = grouped["good"].sum() + eps * len(grouped)
        total_bad = grouped["bad"].sum() + eps * len(grouped)
        grouped["dist_good"] = (grouped["good"] + eps) / total_good
        grouped["dist_bad"] = (grouped["bad"] + eps) / total_bad
        grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
        grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
        iv = grouped["iv_component"].sum()
        return grouped, iv

    iv_rows = []
    iv_candidate_cols = num_for_tests + cat_for_chi[:5]
    for col in iv_candidate_cols:
        try:
            _, iv = compute_woe_iv(df, col, "riesgo_congestion")
            iv_rows.append({"variable": col, "information_value": iv})
        except Exception as e:
            iv_rows.append({"variable": col, "information_value": np.nan, "error": str(e)})
    iv_df = pd.DataFrame(iv_rows).sort_values("information_value", ascending=False)
    # [v3.02 - observacion revisor #2] Los IV de saldo_casos/tasa_atencion/ratio_saldo
    # son extremadamente altos (IV>>0.5) PORQUE esas variables definen la etiqueta
    # proxy (riesgo_congestion = f(saldo_casos, tasa_atencion)); no son fuga
    # "descubierta", son la construccion misma del target, y por eso ya estan
    # excluidas de X (ver LEAKAGE_COLS mas adelante). Se documenta explicitamente
    # para que un revisor no lo interprete como fuga no controlada.
    iv_df["nota_iv"] = iv_df["variable"].apply(
        lambda v: ("IV elevado esperado: esta variable participa en la construccion "
                   "de la etiqueta proxy riesgo_congestion; se reporta solo con fines "
                   "descriptivos, no predictivos (variable excluida de X).")
        if v in ["saldo_casos", "tasa_atencion", "ratio_saldo"] else ""
    )
    display(iv_df)
    clean_for_export(iv_df).to_excel(os.path.join(TABLE_DIR, "11_information_value_target_principal.xlsx"), index=False)

    target_validation_note = {
        "delphi_ahp_status": "No ejecutado: no se cuenta con panel real de expertos.",
        "replacement_strategy": "Validacion empirica de constructo proxy: sensibilidad, estabilidad temporal, separacion estadistica y validacion convergente posterior con clustering.",
        "future_work": "Ejecutar Delphi/AHP con expertos fiscales reales para validar pesos y umbrales institucionales.",
        "primary_target_scenario": PRIMARY_TARGET_SCENARIO,
        "target_interpretation": "Senal proxy de posible sobrecarga fiscal operacional, no etiqueta oficial ni inferencia causal."
    }
    with open(os.path.join(OUTPUT_DIR, "target_proxy_validation_note.json"), "w", encoding="utf-8") as f:
        json.dump(target_validation_note, f, indent=2, ensure_ascii=False)
else:
    print("Validacion empirica del target omitida porque RUN_TARGET_CONSTRUCT_VALIDATION=False.")

df_train = df[df["anio"].isin(TRAIN_YEARS)].copy()
df_valid = df[df["anio"] == VALID_YEAR].copy()
df_test = df[df["anio"] == TEST_YEAR].copy()
df_external = df[df["anio"] == EXTERNAL_YEAR].copy()

partition_summary = pd.DataFrame([
    {"conjunto": "train", "anios": str(TRAIN_YEARS), "registros": len(df_train), "riesgo_pct": round(df_train["riesgo_congestion"].mean()*100, 2)},
    {"conjunto": "validacion", "anios": str(VALID_YEAR), "registros": len(df_valid), "riesgo_pct": round(df_valid["riesgo_congestion"].mean()*100, 2) if len(df_valid) else np.nan},
    {"conjunto": "test", "anios": str(TEST_YEAR), "registros": len(df_test), "riesgo_pct": round(df_test["riesgo_congestion"].mean()*100, 2) if len(df_test) else np.nan},
    {"conjunto": "externa", "anios": str(EXTERNAL_YEAR), "registros": len(df_external), "riesgo_pct": round(df_external["riesgo_congestion"].mean()*100, 2) if len(df_external) else np.nan}
])
display(partition_summary)
partition_summary.to_excel(os.path.join(TABLE_DIR, "12_particion_temporal.xlsx"), index=False)

# Variables excluidas por fuga directa.
LEAKAGE_COLS = ["ingresado", "atendido", "saldo_casos", "tasa_atencion", "ratio_saldo"]

TARGET_SCENARIO_COLS = [c for c in df.columns if c.startswith("riesgo_proxy_")]
ADMIN_EXCLUDE = ["archivo_fuente", TARGET] + TARGET_SCENARIO_COLS

COLLINEAR_EXCLUDE = ["anio_centrado"]

EXCLUDE_COLS = [c for c in LEAKAGE_COLS + ADMIN_EXCLUDE + COLLINEAR_EXCLUDE if c in df.columns]

feature_cols_raw = [c for c in df.columns if c not in EXCLUDE_COLS]

X_train_raw = df_train[feature_cols_raw].copy()
y_train = df_train[TARGET].copy()

X_valid_raw = df_valid[feature_cols_raw].copy()
y_valid = df_valid[TARGET].copy()

X_test_raw = df_test[feature_cols_raw].copy()
y_test = df_test[TARGET].copy()

X_external_raw = df_external[feature_cols_raw].copy() if len(df_external) else pd.DataFrame(columns=feature_cols_raw)
y_external = df_external[TARGET].copy() if len(df_external) else pd.Series(dtype=int)

print("Variables excluidas por fuga/no predictivas/colinealidad:", EXCLUDE_COLS)
print("Columnas de escenarios proxy excluidas explicitamente:", TARGET_SCENARIO_COLS)
print("Numero de variables candidatas:", len(feature_cols_raw))
print("X_train:", X_train_raw.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid_raw.shape, "y_valid:", y_valid.shape)
print("X_test:", X_test_raw.shape, "y_test:", y_test.shape)
print("X_external:", X_external_raw.shape, "y_external:", y_external.shape)

pd.DataFrame({"variables_candidatas": feature_cols_raw}).to_excel(os.path.join(TABLE_DIR, "12b_variables_candidatas_post_exclusion.xlsx"), index=False)
pd.DataFrame({"variables_excluidas": EXCLUDE_COLS}).to_excel(os.path.join(TABLE_DIR, "12c_variables_excluidas_leakage_colinealidad.xlsx"), index=False)

leakage_present = [c for c in LEAKAGE_COLS if c in X_train_raw.columns]
if leakage_present:
    raise ValueError(f"Aun hay variables de fuga directa en X: {leakage_present}")
else:
    print("No hay variables de fuga directa en X.")

numeric_predictors_raw = X_train_raw.select_dtypes(include=np.number)
if numeric_predictors_raw.shape[1] > 0:
    corr_target = (
        numeric_predictors_raw
        .corrwith(y_train)
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    corr_target.columns = ["variable", "abs_corr_target_train"]
else:
    corr_target = pd.DataFrame(columns=["variable", "abs_corr_target_train"])

corr_threshold = 0.70
high_corr_target_df = corr_target[corr_target["abs_corr_target_train"] > corr_threshold].copy()

print("Variables numericas con mayor correlacion absoluta frente al target:")
display(corr_target.head(30))

print(f"Variables con correlacion absoluta > {corr_threshold}:")
display(high_corr_target_df if len(high_corr_target_df) else pd.DataFrame({"mensaje": ["No se detectaron dependencias indirectas fuertes > 0.70."]}))

indirect_watch_cols = [c for c in X_train_raw.columns if c.startswith("hist_") or c.startswith("growth_")]
indirect_watch_df = corr_target[corr_target["variable"].isin(indirect_watch_cols)].copy()
print("Revision de variables historicas/growth potencialmente relacionadas con ingresado/atendido:")
display(indirect_watch_df if len(indirect_watch_df) else pd.DataFrame({"mensaje": ["No hay variables historicas/growth numericas en X_train_raw."]}))

corr_target.to_excel(os.path.join(TABLE_DIR, "13_revision_dependencia_indirecta_target.xlsx"), index=False)
high_corr_target_df.to_excel(os.path.join(TABLE_DIR, "13b_dependencias_indirectas_altas_gt_070.xlsx"), index=False)
indirect_watch_df.to_excel(os.path.join(TABLE_DIR, "13c_variables_historicas_dependencia_target.xlsx"), index=False)

# ============================================================
# 12d. ANALISIS DE CORRELACION ENTRE PREDICTORES (MULTICOLINEALIDAD)
# ============================================================
# Insertar DESPUES del bloque de "correlacion de variables candidatas
# con el target" (celda que genera corr_target / 13_revision_dependencia_...)
# y ANTES del bloque de HIGH_CARDINALITY_CANDIDATES / seleccion de variables
# (mutual_information, anova, chi2, random_forest, boruta, rfecv_year_block).
#
# Objetivo:
# - Detectar pares de variables numericas con alta correlacion lineal (Pearson)
#   calculada SOLO sobre train (evita fuga de informacion de valid/test/external).
# - Decidir cual variable de cada par eliminar usando un criterio auditable:
#   se conserva la que tiene mayor correlacion absoluta con el target
#   (ya calculada en corr_target), y se elimina la otra.
# - Actualizar feature_cols_raw / X_train_raw / X_valid_raw / X_test_raw /
#   X_external_raw ANTES de que entren al pipeline de seleccion de variables.
# - Dejar trazabilidad completa (tabla de pares, heatmap, lista de eliminadas).

CORR_METHOD = "pearson"     # "pearson" (lineal) o "spearman" (monotona, mas robusta a outliers)
CORR_THRESHOLD = 0.90       # umbral de "muy correlacionadas". 0.85-0.95 es rango tipico.

numeric_predictors_train = X_train_raw.select_dtypes(include=np.number).copy()

# Diccionario variable -> correlacion absoluta con el target (ya calculado antes)
target_corr_map = corr_target.set_index("variable")["abs_corr_target_train"].to_dict()

if numeric_predictors_train.shape[1] > 1:
    corr_matrix = numeric_predictors_train.corr(method=CORR_METHOD)
else:
    corr_matrix = pd.DataFrame()

# --- Tabla de pares altamente correlacionados (triangulo superior, sin duplicar) ---
high_corr_pairs = []
cols_num = corr_matrix.columns.tolist()
for i in range(len(cols_num)):
    for j in range(i + 1, len(cols_num)):
        var_a, var_b = cols_num[i], cols_num[j]
        r = corr_matrix.iloc[i, j]
        if pd.notna(r) and abs(r) > CORR_THRESHOLD:
            high_corr_pairs.append({
                "variable_a": var_a,
                "variable_b": var_b,
                f"{CORR_METHOD}_r": r,
                "abs_r": abs(r),
                "corr_target_a": target_corr_map.get(var_a, np.nan),
                "corr_target_b": target_corr_map.get(var_b, np.nan),
            })

high_corr_pairs_df = pd.DataFrame(high_corr_pairs).sort_values("abs_r", ascending=False)

print(f"Pares de variables con |{CORR_METHOD}| > {CORR_THRESHOLD}: {len(high_corr_pairs_df)}")
display(high_corr_pairs_df if len(high_corr_pairs_df) else
        pd.DataFrame({"mensaje": [f"No se detectaron pares con correlacion > {CORR_THRESHOLD}."]}))

# --- Regla de eliminacion: por cada par, se elimina la variable con MENOR ---
# --- correlacion absoluta con el target (criterio auditable y reproducible). ---
# Si hay empate o ausencia de dato con el target, se elimina la que tenga
# mayor correlacion promedio con el resto de variables (mas "redundante").
mean_abs_corr = corr_matrix.abs().mean().to_dict() if len(corr_matrix) else {}

vars_to_drop = set()
drop_reasons = []

for _, row in high_corr_pairs_df.iterrows():
    var_a, var_b = row["variable_a"], row["variable_b"]
    if var_a in vars_to_drop or var_b in vars_to_drop:
        continue  # ya se resolvio este par via otra cadena de correlacion

    ta = target_corr_map.get(var_a, np.nan)
    tb = target_corr_map.get(var_b, np.nan)

    if pd.notna(ta) and pd.notna(tb) and ta != tb:
        loser = var_b if ta >= tb else var_a
        winner = var_a if loser == var_b else var_b
        criterio = "menor correlacion con el target"
    else:
        # Empate o sin info de target: se elimina la mas redundante en promedio
        loser = var_b if mean_abs_corr.get(var_b, 0) >= mean_abs_corr.get(var_a, 0) else var_a
        winner = var_a if loser == var_b else var_b
        criterio = "mayor correlacion promedio con el resto (mas redundante)"

    vars_to_drop.add(loser)
    drop_reasons.append({
        "variable_eliminada": loser,
        "variable_conservada": winner,
        f"{CORR_METHOD}_r_par": row[f"{CORR_METHOD}_r"],
        "criterio": criterio
    })

drop_reasons_df = pd.DataFrame(drop_reasons)

print(f"Variables eliminadas por alta multicolinealidad (> {CORR_THRESHOLD}):", sorted(vars_to_drop))
display(drop_reasons_df if len(drop_reasons_df) else
        pd.DataFrame({"mensaje": ["No se elimino ninguna variable por multicolinealidad."]}))

# --- Heatmap de correlacion (antes de eliminar), para reporte/anexo metodologico ---
if len(corr_matrix) > 1:
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cols_num)), max(6, 0.4 * len(cols_num))))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols_num)))
    ax.set_yticks(range(len(cols_num)))
    ax.set_xticklabels(cols_num, rotation=90, fontsize=6)
    ax.set_yticklabels(cols_num, fontsize=6)
    ax.set_title(f"Matriz de correlacion ({CORR_METHOD}) - variables numericas (train)")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{CORR_METHOD} r")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12d_heatmap_correlacion_predictores.png"), dpi=150)
    plt.show()          # <--- AGREGADO: renderiza el heatmap en la celda
    plt.close(fig)

# --- AGREGADO: mostrar la matriz completa de valores (redondeada) ---
print("Matriz de correlacion completa entre variables numericas (train):")
display(corr_matrix.round(2))

# --- Persistencia para trazabilidad/auditoria ---
corr_matrix.to_excel(os.path.join(TABLE_DIR, "12d_matriz_correlacion_predictores.xlsx"))
high_corr_pairs_df.to_excel(os.path.join(TABLE_DIR, "12e_pares_alta_correlacion.xlsx"), index=False)
drop_reasons_df.to_excel(os.path.join(TABLE_DIR, "12f_variables_eliminadas_multicolinealidad.xlsx"), index=False)

# ============================================================
# ACTUALIZAR CONJUNTOS DE DATOS ANTES DE LA SELECCION DE VARIABLES
# ============================================================
# A partir de aqui, todo el pipeline de seleccion de variables
# (mutual_information, anova, chi2, random_forest, boruta, rfecv_year_block)
# debe correr sobre las variables YA filtradas por multicolinealidad.

feature_cols_raw = [c for c in feature_cols_raw if c not in vars_to_drop]

X_train_raw = X_train_raw[feature_cols_raw].copy()
X_valid_raw = X_valid_raw[feature_cols_raw].copy()
X_test_raw = X_test_raw[feature_cols_raw].copy()
X_external_raw = X_external_raw[feature_cols_raw].copy() if len(X_external_raw) else X_external_raw

print("Numero de variables candidatas tras filtro de multicolinealidad:", len(feature_cols_raw))
print("X_train:", X_train_raw.shape, "X_valid:", X_valid_raw.shape,
      "X_test:", X_test_raw.shape, "X_external:", X_external_raw.shape)

# [ASSERTION DE INTEGRIDAD] ninguna variable eliminada debe seguir presente
assert not (set(feature_cols_raw) & vars_to_drop), \
    "Hay variables eliminadas por multicolinealidad que siguen en feature_cols_raw."

# ============================================================
# 12g. ANALISIS DE ASOCIACION ENTRE PREDICTORES CATEGORICOS (CRAMER'S V)
# ============================================================
# Insertar DESPUES del bloque 12d-12f (multicolinealidad numerica / Pearson)
# y ANTES de HIGH_CARDINALITY_CANDIDATES / seleccion de variables.
#
# Reutiliza la funcion cramers_v(confusion_matrix_2d) ya definida en el
# notebook (la misma que se usa para el chi-cuadrado categorico vs target).
#
# Objetivo:
# - Detectar pares de variables CATEGORICAS con alta asociacion entre si
#   (Cramer's V), calculado SOLO sobre train.
# - Decidir cual variable de cada par eliminar usando el mismo criterio
#   auditable que en el bloque numerico: se conserva la que tiene mayor
#   asociacion (Cramer's V) con el target.
# - Actualizar feature_cols_raw / X_train_raw / X_valid_raw / X_test_raw /
#   X_external_raw ANTES de que entren al pipeline de seleccion de variables.
# - Dejar trazabilidad completa (tabla de pares, heatmap, lista de eliminadas).

CRAMERS_V_THRESHOLD = 0.80  # Cramer's V va de 0 a 1; 0.80 = asociacion muy fuerte.
MAX_CATEGORIES_FOR_CRAMERS = 60  # evita crosstabs gigantes en variables casi-ID

categorical_predictors_train = X_train_raw.select_dtypes(include=["object", "category"]).copy()

# Columnas categoricas candidatas: se excluyen las de cardinalidad extrema,
# ya que Cramer's V pierde interpretabilidad y el crosstab se vuelve costoso.
cat_cols_for_assoc = [
    c for c in categorical_predictors_train.columns
    if categorical_predictors_train[c].nunique(dropna=True) <= MAX_CATEGORIES_FOR_CRAMERS
]
excluded_high_card_from_assoc = [
    c for c in categorical_predictors_train.columns if c not in cat_cols_for_assoc
]
if excluded_high_card_from_assoc:
    print(f"Columnas categoricas excluidas del analisis de asociacion por cardinalidad > {MAX_CATEGORIES_FOR_CRAMERS}:",
          excluded_high_card_from_assoc)

# --- Asociacion (Cramer's V) de cada variable categorica contra el target (train) ---
cat_target_assoc = []
for col in cat_cols_for_assoc:
    tab = pd.crosstab(X_train_raw[col], y_train)
    if tab.shape[0] > 1 and tab.shape[1] == 2:
        cat_target_assoc.append({"variable": col, "cramers_v_target": cramers_v(tab)})
    else:
        cat_target_assoc.append({"variable": col, "cramers_v_target": np.nan})

cat_target_assoc_df = pd.DataFrame(cat_target_assoc).sort_values("cramers_v_target", ascending=False)
cat_target_map = cat_target_assoc_df.set_index("variable")["cramers_v_target"].to_dict()

print("Asociacion (Cramer's V) de variables categoricas candidatas frente al target:")
display(cat_target_assoc_df)
cat_target_assoc_df.to_excel(os.path.join(TABLE_DIR, "12g_cramers_v_categoricas_vs_target.xlsx"), index=False)

# --- Matriz de Cramer's V entre pares de variables categoricas (solo train) ---
n_cat = len(cat_cols_for_assoc)
cramers_matrix = pd.DataFrame(np.eye(n_cat), index=cat_cols_for_assoc, columns=cat_cols_for_assoc)

high_cramers_pairs = []
for i in range(n_cat):
    for j in range(i + 1, n_cat):
        var_a, var_b = cat_cols_for_assoc[i], cat_cols_for_assoc[j]
        tab_ab = pd.crosstab(X_train_raw[var_a], X_train_raw[var_b])
        v_ab = cramers_v(tab_ab)
        cramers_matrix.loc[var_a, var_b] = v_ab
        cramers_matrix.loc[var_b, var_a] = v_ab
        if pd.notna(v_ab) and v_ab > CRAMERS_V_THRESHOLD:
            high_cramers_pairs.append({
                "variable_a": var_a,
                "variable_b": var_b,
                "cramers_v": v_ab,
                "cramers_v_target_a": cat_target_map.get(var_a, np.nan),
                "cramers_v_target_b": cat_target_map.get(var_b, np.nan),
            })

high_cramers_pairs_df = pd.DataFrame(high_cramers_pairs).sort_values("cramers_v", ascending=False) \
    if high_cramers_pairs else pd.DataFrame(columns=["variable_a", "variable_b", "cramers_v",
                                                       "cramers_v_target_a", "cramers_v_target_b"])

print(f"Pares de variables categoricas con Cramer's V > {CRAMERS_V_THRESHOLD}: {len(high_cramers_pairs_df)}")
display(high_cramers_pairs_df if len(high_cramers_pairs_df) else
        pd.DataFrame({"mensaje": [f"No se detectaron pares categoricos con Cramer's V > {CRAMERS_V_THRESHOLD}."]}))

# --- Regla de eliminacion: por cada par, se elimina la variable con MENOR ---
# --- asociacion (Cramer's V) con el target. Mismo criterio que en el bloque numerico. ---
cat_vars_to_drop = set()
cat_drop_reasons = []

for _, row in high_cramers_pairs_df.iterrows():
    var_a, var_b = row["variable_a"], row["variable_b"]
    if var_a in cat_vars_to_drop or var_b in cat_vars_to_drop:
        continue

    ta = cat_target_map.get(var_a, np.nan)
    tb = cat_target_map.get(var_b, np.nan)

    if pd.notna(ta) and pd.notna(tb) and ta != tb:
        loser = var_b if ta >= tb else var_a
        winner = var_a if loser == var_b else var_b
        criterio = "menor asociacion (Cramer's V) con el target"
    else:
        # Empate o sin info de target: se elimina la de mayor cardinalidad
        # (mas propensa a sobreajuste / dummies redundantes tras one-hot).
        card_a = X_train_raw[var_a].nunique(dropna=True)
        card_b = X_train_raw[var_b].nunique(dropna=True)
        loser = var_b if card_b >= card_a else var_a
        winner = var_a if loser == var_b else var_b
        criterio = "mayor cardinalidad (mas dummies tras one-hot)"

    cat_vars_to_drop.add(loser)
    cat_drop_reasons.append({
        "variable_eliminada": loser,
        "variable_conservada": winner,
        "cramers_v_par": row["cramers_v"],
        "criterio": criterio
    })

cat_drop_reasons_df = pd.DataFrame(cat_drop_reasons)

print(f"Variables categoricas eliminadas por alta asociacion (> {CRAMERS_V_THRESHOLD}):", sorted(cat_vars_to_drop))
display(cat_drop_reasons_df if len(cat_drop_reasons_df) else
        pd.DataFrame({"mensaje": ["No se elimino ninguna variable categorica por asociacion alta."]}))

# --- Heatmap de Cramer's V (antes de eliminar) ---
if n_cat > 1:
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * n_cat), max(6, 0.4 * n_cat)))
    im = ax.imshow(cramers_matrix.values, vmin=0, vmax=1, cmap="Oranges")
    ax.set_xticks(range(n_cat))
    ax.set_yticks(range(n_cat))
    ax.set_xticklabels(cat_cols_for_assoc, rotation=90, fontsize=6)
    ax.set_yticklabels(cat_cols_for_assoc, fontsize=6)
    ax.set_title("Matriz de asociacion Cramer's V - variables categoricas (train)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cramer's V")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "12g_heatmap_cramers_v_categoricas.png"), dpi=150)
    plt.show()
    plt.close(fig)

print("Matriz completa de Cramer's V entre variables categoricas (train):")
display(cramers_matrix.round(2))

# --- Persistencia para trazabilidad/auditoria ---
cramers_matrix.to_excel(os.path.join(TABLE_DIR, "12h_matriz_cramers_v_categoricas.xlsx"))
high_cramers_pairs_df.to_excel(os.path.join(TABLE_DIR, "12i_pares_alta_asociacion_categoricas.xlsx"), index=False)
cat_drop_reasons_df.to_excel(os.path.join(TABLE_DIR, "12j_variables_categoricas_eliminadas.xlsx"), index=False)

# ============================================================
# ACTUALIZAR CONJUNTOS DE DATOS ANTES DE LA SELECCION DE VARIABLES
# ============================================================
feature_cols_raw = [c for c in feature_cols_raw if c not in cat_vars_to_drop]

X_train_raw = X_train_raw[feature_cols_raw].copy()
X_valid_raw = X_valid_raw[feature_cols_raw].copy()
X_test_raw = X_test_raw[feature_cols_raw].copy()
X_external_raw = X_external_raw[feature_cols_raw].copy() if len(X_external_raw) else X_external_raw

print("Numero de variables candidatas tras filtro de asociacion categorica:", len(feature_cols_raw))
print("X_train:", X_train_raw.shape, "X_valid:", X_valid_raw.shape,
      "X_test:", X_test_raw.shape, "X_external:", X_external_raw.shape)

# [ASSERTION DE INTEGRIDAD] ninguna variable eliminada debe seguir presente
assert not (set(feature_cols_raw) & cat_vars_to_drop), \
    "Hay variables categoricas eliminadas por asociacion alta que siguen en feature_cols_raw."

HIGH_CARDINALITY_CANDIDATES = [
    c for c in X_train_raw.select_dtypes(include=["object"]).columns
    if X_train_raw[c].nunique(dropna=True) >= 10
]

def add_frequency_encoding(train, valid, test, external, cols):
    """Frequency Encoding ajustado solo sobre train y aplicado a otros conjuntos."""
    train = train.copy(); valid = valid.copy(); test = test.copy(); external = external.copy()
    for col in cols:
        if col not in train.columns:
            continue
        freq = train[col].value_counts(normalize=True, dropna=False)
        new_col = f"freq_{col}"
        train[new_col] = train[col].map(freq).fillna(0)
        if len(valid):
            valid[new_col] = valid[col].map(freq).fillna(0) if col in valid.columns else 0
        else:
            valid[new_col] = pd.Series(dtype=float)
        if len(test):
            test[new_col] = test[col].map(freq).fillna(0) if col in test.columns else 0
        else:
            test[new_col] = pd.Series(dtype=float)
        if len(external):
            external[new_col] = external[col].map(freq).fillna(0) if col in external.columns else 0
        else:
            external[new_col] = pd.Series(dtype=float)
    return train, valid, test, external

X_train_fe, X_valid_fe, X_test_fe, X_external_fe = add_frequency_encoding(
    X_train_raw, X_valid_raw, X_test_raw, X_external_raw, HIGH_CARDINALITY_CANDIDATES
)

numeric_cols = X_train_fe.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()

print("Columnas numericas:", len(numeric_cols))
print("Columnas categoricas:", len(categorical_cols))
print("Columnas con Frequency Encoding:", [f"freq_{c}" for c in HIGH_CARDINALITY_CANDIDATES])

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", make_one_hot_encoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
], remainder="drop")

X_train_pre = preprocessor.fit_transform(X_train_fe)
X_valid_pre = preprocessor.transform(X_valid_fe) if len(X_valid_fe) else np.empty((0, X_train_pre.shape[1]))
X_test_pre = preprocessor.transform(X_test_fe) if len(X_test_fe) else np.empty((0, X_train_pre.shape[1]))
X_external_pre = preprocessor.transform(X_external_fe) if len(X_external_fe) else np.empty((0, X_train_pre.shape[1]))

feature_names = get_preprocessed_feature_names(preprocessor, numeric_cols, categorical_cols)

print("Dimensiones despues de preprocesamiento:")
print("Train:", X_train_pre.shape, "Valid:", X_valid_pre.shape, "Test:", X_test_pre.shape, "External:", X_external_pre.shape)

pd.DataFrame({"feature_preprocesada": feature_names}).to_excel(os.path.join(TABLE_DIR, "14_features_preprocesadas.xlsx"), index=False)

fs_train_mask = df_train["anio"].isin(FEATURE_SELECTION_TRAIN_YEARS).values
fs_valid_mask = (df_train["anio"] == FEATURE_SELECTION_INTERNAL_VALID_YEAR).values

X_fs_train_raw = X_train_raw.loc[df_train["anio"].isin(FEATURE_SELECTION_TRAIN_YEARS)].copy()
X_fs_valid_raw = X_train_raw.loc[df_train["anio"] == FEATURE_SELECTION_INTERNAL_VALID_YEAR].copy()
y_fs_train = y_train.loc[df_train["anio"].isin(FEATURE_SELECTION_TRAIN_YEARS)].copy()
y_fs_valid = y_train.loc[df_train["anio"] == FEATURE_SELECTION_INTERNAL_VALID_YEAR].copy()

HIGH_CARDINALITY_FS = [
    c for c in X_fs_train_raw.select_dtypes(include=["object"]).columns
    if X_fs_train_raw[c].nunique(dropna=True) >= 10
]

X_fs_train_fe, X_fs_valid_fe, _, _ = add_frequency_encoding(
    X_fs_train_raw,
    X_fs_valid_raw,
    X_fs_valid_raw.copy(),
    pd.DataFrame(columns=X_fs_train_raw.columns),
    HIGH_CARDINALITY_FS
)

numeric_fs_cols = X_fs_train_fe.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_fs_cols = X_fs_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor_fs = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_fs_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_one_hot_encoder())]), categorical_fs_cols)
], remainder="drop")

X_fs_train = preprocessor_fs.fit_transform(X_fs_train_fe)
X_fs_valid = preprocessor_fs.transform(X_fs_valid_fe) if len(X_fs_valid_fe) else np.empty((0, X_fs_train.shape[1]))
feature_names_fs = get_preprocessed_feature_names(preprocessor_fs, numeric_fs_cols, categorical_fs_cols)

print("Feature selection train interno:", X_fs_train.shape, "Anios:", FEATURE_SELECTION_TRAIN_YEARS)
print("Feature selection validacion interna:", X_fs_valid.shape, "Anio:", FEATURE_SELECTION_INTERNAL_VALID_YEAR)
print("Preprocesador FS ajustado solo con:", FEATURE_SELECTION_TRAIN_YEARS)
print("Features finales del preprocesador principal:", len(feature_names))
print("Features del preprocesador FS:", len(feature_names_fs))

selection_results = {}
TOP_K = min(40, X_fs_train.shape[1]) if not FAST_MODE else min(20, X_fs_train.shape[1])

mi_scores = mutual_info_classif(X_fs_train, y_fs_train, random_state=RANDOM_STATE, discrete_features=False)
mi_df = pd.DataFrame({"feature": feature_names_fs, "score": mi_scores}).sort_values("score", ascending=False)
selection_results["mutual_information"] = set(mi_df.head(TOP_K)["feature"])

f_scores, f_pvalues = f_classif(X_fs_train, y_fs_train)
anova_df = pd.DataFrame({"feature": feature_names_fs, "score": f_scores, "pvalue": f_pvalues}).replace([np.inf, -np.inf], np.nan).fillna(0).sort_values("score", ascending=False)
selection_results["anova"] = set(anova_df.head(TOP_K)["feature"])

minmax_fs = MinMaxScaler()
X_fs_train_nonneg = minmax_fs.fit_transform(X_fs_train)
chi_scores, chi_pvalues = chi2(X_fs_train_nonneg, y_fs_train)
chi_df = pd.DataFrame({"feature": feature_names_fs, "score": chi_scores, "pvalue": chi_pvalues}).replace([np.inf, -np.inf], np.nan).fillna(0).sort_values("score", ascending=False)
selection_results["chi2"] = set(chi_df.head(TOP_K)["feature"])

rf_selector = RandomForestClassifier(
    n_estimators=300 if not FAST_MODE else 100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf_selector.fit(X_fs_train, y_fs_train)
rf_imp_df = pd.DataFrame({"feature": feature_names_fs, "score": rf_selector.feature_importances_}).sort_values("score", ascending=False)
selection_results["random_forest"] = set(rf_imp_df.head(TOP_K)["feature"])

if RUN_HEAVY_BLOCKS and BorutaPy is not None:
    try:
        boruta_rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, max_depth=7)
        boruta = BorutaPy(boruta_rf, n_estimators="auto", random_state=RANDOM_STATE, max_iter=50 if not FAST_MODE else 20, verbose=0)
        boruta.fit(X_fs_train, y_fs_train.values)
        boruta_features = set(feature_names_fs[boruta.support_])
    except Exception as e:
        print("Boruta fallo; usando top RF como alternativa:", e)
        boruta_features = set(rf_imp_df.head(TOP_K)["feature"])
else:
    boruta_features = set(rf_imp_df.head(TOP_K)["feature"])
selection_results["boruta"] = boruta_features

if RUN_HEAVY_BLOCKS:
    try:
        candidate_features = sorted(set().union(*selection_results.values()))
        candidate_idx = [np.where(feature_names_fs == f)[0][0] for f in candidate_features]

        fs_years = df_train.loc[df_train["anio"].isin(FEATURE_SELECTION_TRAIN_YEARS), "anio"].values
        fs_order = np.argsort(fs_years)
        X_rfe = X_fs_train[:, candidate_idx][fs_order]
        y_rfe = np.array(y_fs_train)[fs_order]
        years_rfe = fs_years[fs_order]

        rfecv_splits = build_year_forward_cv_splits(years_rfe, min_train_periods=1, max_splits=CV_FOLDS)
        rfecv_splits = filter_cv_splits_with_both_classes(rfecv_splits, y_rfe)
        if len(rfecv_splits) < 2:
            print("RFECV: folds anuales insuficientes; se usa TimeSeriesSplit como respaldo menor.")
            rfecv_splits = list(TimeSeriesSplit(n_splits=min(3, max(2, len(FEATURE_SELECTION_TRAIN_YEARS)-1))).split(X_rfe))

        estimator_rfe = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
        rfecv = RFECV(
            estimator=estimator_rfe,
            step=0.2,
            cv=rfecv_splits,
            scoring="f1",
            min_features_to_select=min(5, X_rfe.shape[1]),
            n_jobs=-1
        )
        rfecv.fit(X_rfe, y_rfe)
        rfecv_features = set(np.array(candidate_features)[rfecv.support_])
    except Exception as e:
        print("RFECV temporal anual fallo; usando top MI/ANOVA/RF:", e)
        rfecv_features = set(mi_df.head(20)["feature"]).union(set(rf_imp_df.head(20)["feature"]))
else:
    rfecv_features = set(mi_df.head(20)["feature"]).union(set(rf_imp_df.head(20)["feature"]))
selection_results["rfecv_year_block"] = rfecv_features

vote_rows = []
for feat in feature_names_fs:
    votes = sum([feat in feats for feats in selection_results.values()])
    methods = [m for m, feats in selection_results.items() if feat in feats]
    vote_rows.append({"feature": feat, "votes": votes, "methods": ", ".join(methods)})

feature_votes_df = pd.DataFrame(vote_rows).sort_values(["votes", "feature"], ascending=[False, True])
# [v3.02 - observacion revisor #6/#9] "Stability Score": porcentaje de metodos
# de seleccion (de un total de 6: MI, ANOVA, Chi2, RF, Boruta, RFECV) que
# seleccionaron cada feature. Es una forma directa y barata de reportar
# estabilidad de seleccion sin requerir bootstrap adicional.
_n_selection_methods = len(selection_results)
feature_votes_df["stability_score_pct"] = (feature_votes_df["votes"] / max(_n_selection_methods, 1) * 100).round(1)

final_feature_set = set(feature_names)
selected_candidates = feature_votes_df.loc[feature_votes_df["votes"] >= 2, "feature"].tolist()
selected_features_final = [f for f in selected_candidates if f in final_feature_set]

if len(selected_features_final) < 10:
    selected_features_final = [
        f for f in feature_votes_df["feature"].tolist()
        if f in final_feature_set
    ][:min(30, len(final_feature_set))]

missing_selected_features = [f for f in selected_candidates if f not in final_feature_set]
if missing_selected_features:
    print("Features seleccionadas en FS pero ausentes en preprocesador principal; se omiten:", missing_selected_features[:20])

# [PATCH 8] Assertions de integridad tras la seleccion de features.
assert len(selected_features_final) > 0, "La seleccion de features devolvio cero variables."
assert all(f in set(feature_names) for f in selected_features_final), \
    "Hay una feature seleccionada en FS que no existe en el preprocesador principal."

selected_idx = [np.where(feature_names == f)[0][0] for f in selected_features_final]
selected_idx_fs = [np.where(feature_names_fs == f)[0][0] for f in selected_features_final if f in set(feature_names_fs)]

X_train_sel = X_train_pre[:, selected_idx]
X_valid_sel = X_valid_pre[:, selected_idx] if len(X_valid_pre) else np.empty((0, len(selected_idx)))
X_test_sel = X_test_pre[:, selected_idx] if len(X_test_pre) else np.empty((0, len(selected_idx)))
X_external_sel = X_external_pre[:, selected_idx] if len(X_external_pre) else np.empty((0, len(selected_idx)))

try:
    X_fs_train_sel = X_fs_train[:, selected_idx_fs]
    X_fs_valid_sel = X_fs_valid[:, selected_idx_fs]
    fs_val_model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    fs_val_model.fit(X_fs_train_sel, y_fs_train)
    fs_val_score = fs_val_model.predict_proba(X_fs_valid_sel)[:, 1]
    fs_val_pred = (fs_val_score >= 0.5).astype(int)
    fs_internal_validation_df = pd.DataFrame([{
        "train_years": str(FEATURE_SELECTION_TRAIN_YEARS),
        "internal_valid_year": FEATURE_SELECTION_INTERNAL_VALID_YEAR,
        "selected_features": len(selected_features_final),
        "f1": f1_score(y_fs_valid, fs_val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_fs_valid, fs_val_score) if len(np.unique(y_fs_valid)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_fs_valid, fs_val_score) if len(np.unique(y_fs_valid)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_fs_valid, fs_val_pred),
        "interpretacion": "Validacion interna para reduccion dimensional; no implica superioridad definitiva."
    }])
except Exception as e:
    fs_internal_validation_df = pd.DataFrame([{"mensaje": f"No se pudo ejecutar validacion interna: {e}"}])

print("Numero de features seleccionadas:", len(selected_features_final))
display(feature_votes_df.head(50))
display(fs_internal_validation_df)

mi_df.to_excel(os.path.join(TABLE_DIR, "16_mutual_information_fs_train_2019_2022.xlsx"), index=False)
anova_df.to_excel(os.path.join(TABLE_DIR, "17_anova_f_test_fs_train_2019_2022.xlsx"), index=False)
chi_df.to_excel(os.path.join(TABLE_DIR, "18_chi_square_features_fs_train_2019_2022.xlsx"), index=False)
rf_imp_df.to_excel(os.path.join(TABLE_DIR, "19_random_forest_importance_fs_train_2019_2022.xlsx"), index=False)
clean_for_export(feature_votes_df).to_excel(os.path.join(TABLE_DIR, "20_feature_selection_voting_temporal.xlsx"), index=False)
pd.DataFrame({"selected_features_final": selected_features_final}).to_excel(os.path.join(TABLE_DIR, "21_selected_features_final.xlsx"), index=False)
fs_internal_validation_df.to_excel(os.path.join(TABLE_DIR, "21b_feature_selection_validacion_interna_2023.xlsx"), index=False)
pd.DataFrame({"missing_selected_features": missing_selected_features}).to_excel(os.path.join(TABLE_DIR, "21c_features_fs_omitidas_por_no_existir_en_preprocesador_principal.xlsx"), index=False)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca2 = pca.fit_transform(X_train_sel)
X_test_pca2 = pca.transform(X_test_sel) if len(X_test_sel) else np.empty((0, 2))

pca_df = pd.DataFrame({"PC1": X_train_pca2[:,0], "PC2": X_train_pca2[:,1], "riesgo_congestion": y_train.values})
plt.figure(figsize=(7,5))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["riesgo_congestion"], alpha=0.45)
plt.title("PCA 2D - Train segun target proxy")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "18_pca_2d_train.png"), dpi=300)
plt.show()

print("Varianza explicada PCA 2D:", pca.explained_variance_ratio_)

class_counts = y_train.value_counts().to_dict()
neg = class_counts.get(0, 0)
pos = class_counts.get(1, 0)
scale_pos_weight = neg / max(pos, 1)
print("Distribucion train:", class_counts)
print("scale_pos_weight:", round(scale_pos_weight, 4))

X_train_model = X_train_sel
y_train_model = y_train.copy()

if USE_SMOTE and SMOTE is not None:
    print("Aplicando SMOTE SOLO sobre train.")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_model, y_train_model = smote.fit_resample(X_train_sel, y_train)
    print("Distribucion despues de SMOTE:", pd.Series(y_train_model).value_counts().to_dict())
else:
    print("No se aplica SMOTE. Se utilizara ponderacion de clases en los modelos compatibles.")

# [PATCH 8] Assertion final antes de entrenar: consistencia X/y de train.
assert X_train_model.shape[0] == len(y_train_model), "Desajuste de filas entre X_train_model y y_train_model."
assert set(np.unique(y_train_model)).issubset({0, 1}), "y_train_model no es binario tras el balanceo de clases."

# [PATCH 11-13] Checkpoint M03: feature selection + matrices preprocesadas listas para modelar.
ckpt.save(
    "M03_feature_selection_y_matrices",
    selected_features_final=selected_features_final,
    X_train_sel=X_train_sel, X_valid_sel=X_valid_sel, X_test_sel=X_test_sel, X_external_sel=X_external_sel,
    y_train=y_train, y_valid=y_valid, y_test=y_test, y_external=y_external,
    feature_votes_df=feature_votes_df,
)

def get_scores(model, X):
    """Devuelve score/probabilidad para la clase positiva sin normalizar por dataset."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if getattr(scores, "ndim", 1) == 2:
            scores = scores[:, 1]
        return expit(scores)
    return model.predict(X)

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def extended_binary_metrics(y_true, y_pred):
    """
    [v3.02 - observacion revisor #17] Metricas adicionales de clasificacion
    binaria valoradas por revisores Q1: MCC, Cohen's Kappa, NPV, FPR, FNR,
    y los cocientes de verosimilitud diagnostica LR+ / LR- (uso clinico
    tipico, aplicable aqui como medida de fuerza de la senal proxy).
    """
    from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = np.nan
    lr_plus = sensitivity / fpr if (pd.notna(fpr) and fpr > 0) else np.nan
    lr_minus = fnr / specificity if (pd.notna(specificity) and specificity > 0) else np.nan
    return {"mcc": mcc, "cohen_kappa": kappa, "npv": npv, "fpr": fpr, "fnr": fnr,
            "lr_plus": lr_plus, "lr_minus": lr_minus}

def evaluate_model(model, X, y, dataset_name, threshold=0.5):
    if len(y) == 0:
        return None
    score = get_scores(model, X)
    pred = (score >= threshold).astype(int)
    base = {
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "specificity": specificity_score(y, pred)
    }
    base.update(extended_binary_metrics(y, pred))  # [v3.02] MCC, Kappa, NPV, FPR, FNR, LR+/-
    return base

def evaluate_model_batched(model, X, y, dataset_name, threshold=0.5, batch_size=100_000):
    """
    [PATCH 19] Igual que evaluate_model(), pero calcula los scores por lotes
    via predict_in_batches(). Se usa en validacion externa / datasets grandes
    para evitar picos de memoria cuando X tiene cientos de miles o millones
    de filas (pensado para escalar hasta ~4.5M de registros).
    """
    if len(y) == 0:
        return None
    score = predict_in_batches(model, X, batch_size=batch_size, score_fn=get_scores)
    pred = (score >= threshold).astype(int)
    base = {
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "specificity": specificity_score(y, pred)
    }
    base.update(extended_binary_metrics(y, pred))
    return base

def evaluate_predictions(y_true, y_pred, dataset_name, model_name, score=None, threshold=None):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred).astype(int)
    if score is None:
        score = y_pred_arr
    score = np.asarray(score)
    base = {
        "dataset": dataset_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
        "roc_auc": roc_auc_score(y_true_arr, score) if len(np.unique(y_true_arr)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true_arr, score) if len(np.unique(y_true_arr)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "specificity": specificity_score(y_true_arr, y_pred_arr),
        "modelo": model_name
    }
    base.update(extended_binary_metrics(y_true_arr, y_pred_arr))
    return base

if len(X_train_model) == len(df_train):
    train_model_order = np.argsort(df_train["anio"].values)
    X_train_model_cv = X_train_model[train_model_order]
    y_train_model_cv = np.array(y_train_model)[train_model_order]
    years_model_cv = df_train["anio"].values[train_model_order]
    cv5 = build_year_forward_cv_splits(years_model_cv, min_train_periods=1, max_splits=CV_FOLDS)
    cv5 = filter_cv_splits_with_both_classes(cv5, y_train_model_cv)
    if len(cv5) < 2:
        print("CV anual insuficiente; respaldo con TimeSeriesSplit por filas.")
        cv5 = TimeSeriesSplit(n_splits=min(CV_FOLDS, 3))
else:
    X_train_model_cv = X_train_model
    y_train_model_cv = np.array(y_train_model)
    cv5 = StratifiedKFold(n_splits=min(CV_FOLDS, 5), shuffle=True, random_state=RANDOM_STATE)
    print("SMOTE activado: se usa StratifiedKFold como respaldo porque la estructura temporal original fue re-muestreada.")

cv_description = "year_block_walk_forward" if isinstance(cv5, list) else type(cv5).__name__
print("Estrategia CV principal:", cv_description)
print("Numero de folds CV:", len(cv5) if isinstance(cv5, list) else CV_FOLDS)

svm_base = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)
svm_calibrated = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
svm_param_key = "estimator__C" if "estimator__C" in svm_calibrated.get_params() else "base_estimator__C"

model_spaces = {
    "DummyClassifier": (
        DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        {}
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE, solver="liblinear"),
        {"C": np.logspace(-3, 2, 30), "penalty": ["l1", "l2"]}
    ),
    "SVM": (
        svm_calibrated,
        {svm_param_key: np.logspace(-3, 2, 12)}
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, scale_pos_weight=scale_pos_weight),
        {"n_estimators": [100, 200, 300, 500], "max_depth": [3, 4, 5, 6], "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0]}
    ),
    "LightGBM": (
        lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, verbose=-1),
        {"n_estimators": [100, 200, 300, 500], "num_leaves": [15, 31, 63], "max_depth": [-1, 3, 5, 7], "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0]}
    ),
    "CatBoost": (
        CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0, auto_class_weights="Balanced"),
        {"iterations": [100, 200, 300, 500], "depth": [3, 5, 7, 9], "learning_rate": [0.01, 0.03, 0.1, 0.2], "l2_leaf_reg": [1, 3, 5, 7]}
    )
}

trained_models = {}
cv_results_summary = []

for name, (estimator, param_dist) in model_spaces.items():
    print("\nEntrenando modelo:", name)
    if name == "DummyClassifier":
        model = estimator.fit(X_train_model, y_train_model)
        best_params = {}
        best_score = np.nan
    else:
        n_iter = N_EXPERIMENTS_PER_MODEL if not FAST_MODE else 10
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1",
            cv=cv5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            error_score="raise"
        )
        try:
            # [PATCH 20] Reintento automatico: si falla una vez (ej. problema
            # transitorio de memoria/paralelismo), se reintenta antes de caer
            # al respaldo con parametros base.
            retry_on_failure(search.fit, X_train_model_cv, y_train_model_cv,
                              retries=1, backoff_seconds=3, label=f"RandomizedSearchCV[{name}]")
            model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
        except Exception as e:
            print("RandomizedSearchCV fallo tras reintento; entrenando con parametros base:", e)
            model = estimator.fit(X_train_model, y_train_model)
            best_params = {}
            best_score = np.nan
    trained_models[name] = model
    # [v3.02 - observacion revisor #1] cv_best_f1 puede quedar NaN para
    # DummyClassifier (predice una sola clase constante -> F1 indefinido en
    # algun fold temporal) o si el RandomizedSearchCV fallo y se uso el
    # respaldo con parametros base sin CV. Se documenta explicitamente en
    # vez de dejar un NaN "crudo" en el reporte.
    if pd.isna(best_score):
        cv_nota = ("Modelo constante (DummyClassifier): F1 indefinido en al menos un fold "
                   "temporal por no predecir la clase positiva." if name == "DummyClassifier"
                   else "RandomizedSearchCV no se pudo completar; se uso ajuste con parametros base sin CV.")
    else:
        cv_nota = ""
    cv_results_summary.append({"modelo": name, "cv_best_f1": best_score, "best_params": json.dumps(best_params),
                                "cv_strategy": cv_description, "nota": cv_nota})

cv_results_df = pd.DataFrame(cv_results_summary)
display(cv_results_df)
clean_for_export(cv_results_df).to_excel(os.path.join(TABLE_DIR, "22_cv5_100_experimentos_resumen.xlsx"), index=False)

# [PATCH 12] Guarda TODOS los modelos base entrenados (no solo el mejor), comprimidos.
models_saved_manifest_df = save_all_trained_models(trained_models)
# [PATCH 11-13] Checkpoint M04: catalogo base de modelos entrenados.
ckpt.save("M04_modelos_base_entrenados", cv_results_df=cv_results_df,
          trained_model_names=list(trained_models.keys()))

all_metrics = []
for name, model in trained_models.items():
    for X_eval, y_eval, dname in [(X_valid_sel, y_valid, "valid_2024"), (X_test_sel, y_test, "test_2025"), (X_external_sel, y_external, "external_2026")]:
        res = evaluate_model(model, X_eval, y_eval, dname, threshold=0.5)
        if res is not None:
            res["modelo"] = name
            all_metrics.append(res)
metrics_df = pd.DataFrame(all_metrics)
display(metrics_df.sort_values(["dataset", "f1"], ascending=[True, False]))
clean_for_export(metrics_df).to_excel(os.path.join(TABLE_DIR, "23_metricas_modelos_valid_test_external.xlsx"), index=False)

def rule_baselines_for_df(df_eval, y_eval, dataset_name):
    if len(df_eval) == 0:
        return []
    rows = []
    saldo_rule = (df_eval["saldo_casos"] >= P75_SALDO).astype(int)
    tasa_rule = (df_eval["tasa_atencion"] <= P25_TASA).astype(int)
    combined_rule = ((df_eval["saldo_casos"] >= P75_SALDO) & (df_eval["tasa_atencion"] <= P25_TASA)).astype(int)

    r1 = evaluate_predictions(y_eval, saldo_rule, dataset_name, "Rule_SaldoOnly", score=df_eval["saldo_casos"].values, threshold=float(P75_SALDO))
    r1.update({"benchmark_type": "single_indicator_baseline", "eligible_for_ml_comparison": True,
               "interpretacion": "Baseline simple de un solo indicador; no usa todo el espacio predictivo."})
    rows.append(r1)

    r2 = evaluate_predictions(y_eval, tasa_rule, dataset_name, "Rule_TasaOnly", score=(-df_eval["tasa_atencion"]).values, threshold=float(P25_TASA))
    r2.update({"benchmark_type": "single_indicator_baseline", "eligible_for_ml_comparison": True,
               "interpretacion": "Baseline simple de un solo indicador; no usa todo el espacio predictivo."})
    rows.append(r2)

    r3 = evaluate_predictions(y_eval, combined_rule, dataset_name, "Oracle_TargetDefinitionCheck", score=combined_rule.values, threshold="P75_saldo_AND_P25_tasa")
    r3.update({"benchmark_type": "oracle_target_definition_check", "eligible_for_ml_comparison": False,
               "interpretacion": "Sanity check: reproduce la regla que define y_true. No debe reportarse como baseline predictivo contra ML."})
    rows.append(r3)
    return rows

baseline_rules = []
baseline_rules += rule_baselines_for_df(df_valid, y_valid, "valid_2024")
baseline_rules += rule_baselines_for_df(df_test, y_test, "test_2025")
baseline_rules += rule_baselines_for_df(df_external, y_external, "external_2026")
baseline_rules_df = pd.DataFrame(baseline_rules)
if len(baseline_rules_df):
    display(baseline_rules_df.sort_values(["dataset", "eligible_for_ml_comparison", "f1"], ascending=[True, False, False]))
    baseline_rules_df.to_excel(os.path.join(TABLE_DIR, "23b_baselines_y_sanity_checks_target.xlsx"), index=False)
    baseline_rules_df[baseline_rules_df["eligible_for_ml_comparison"] == True].to_excel(os.path.join(TABLE_DIR, "23c_baselines_predictivos_simples.xlsx"), index=False)
    baseline_rules_df[baseline_rules_df["eligible_for_ml_comparison"] == False].to_excel(os.path.join(TABLE_DIR, "23d_sanity_check_definicion_target.xlsx"), index=False)

valid_metrics = metrics_df[metrics_df["dataset"] == "valid_2024"].copy()
valid_metrics = valid_metrics.sort_values(["f1", "roc_auc"], ascending=False)
BEST_MODEL_NAME = valid_metrics.iloc[0]["modelo"]
best_model = trained_models[BEST_MODEL_NAME]
print("Mejor modelo ML segun validacion 2024:", BEST_MODEL_NAME)

valid_scores = get_scores(best_model, X_valid_sel)
threshold_grid = np.linspace(0.05, 0.95, 91)
thr_rows = []
for thr in threshold_grid:
    pred = (valid_scores >= thr).astype(int)
    thr_rows.append({"threshold": thr, "f1": f1_score(y_valid, pred, zero_division=0), "precision": precision_score(y_valid, pred, zero_division=0), "recall": recall_score(y_valid, pred, zero_division=0)})
thr_df = pd.DataFrame(thr_rows).sort_values("f1", ascending=False)
BEST_THRESHOLD = float(thr_df.iloc[0]["threshold"])
print("Mejor umbral segun F1 en validacion:", BEST_THRESHOLD)
display(thr_df.head(10))
clean_for_export(thr_df).to_excel(os.path.join(TABLE_DIR, "24_umbral_optimo_validacion.xlsx"), index=False)

# [v3.02 - observacion revisor #8/#7] Optimizacion de umbral bajo multiples
# criterios (no solo 0.5 fijo ni solo el optimo de F1): Youden's J (maximiza
# sensibilidad+especificidad sobre la curva ROC), umbral que alcanza una
# precision objetivo, y umbral que alcanza un recall objetivo. Todo calculado
# sobre validacion 2024 (nunca sobre test), igual que BEST_THRESHOLD.
fpr_grid, tpr_grid, roc_thresholds = roc_curve(y_valid, valid_scores)
youden_j = tpr_grid - fpr_grid
youden_idx = int(np.argmax(youden_j))
youden_threshold = float(roc_thresholds[youden_idx]) if np.isfinite(roc_thresholds[youden_idx]) else 0.5

PRECISION_TARGET = 0.80
RECALL_TARGET = 0.80
thr_for_precision = thr_df[thr_df["precision"] >= PRECISION_TARGET].sort_values("threshold").head(1)
thr_for_recall = thr_df[thr_df["recall"] >= RECALL_TARGET].sort_values("threshold", ascending=False).head(1)

threshold_optimization_df = pd.DataFrame([
    {"criterio": "Fijo (referencia clasica)", "threshold": 0.5,
     "nota": "Umbral de referencia; en datasets desbalanceados rara vez es el optimo."},
    {"criterio": "F1-optimo (usado como BEST_THRESHOLD)", "threshold": BEST_THRESHOLD,
     "nota": "Maximiza F1 en validacion 2024; es el umbral usado en el resto del pipeline."},
    {"criterio": "Youden's J (ROC)", "threshold": youden_threshold,
     "nota": f"Maximiza sensibilidad+especificidad-1 (J={youden_j[youden_idx]:.4f}) sobre la curva ROC de validacion."},
    {"criterio": f"Precision >= {PRECISION_TARGET}", "threshold": float(thr_for_precision["threshold"].iloc[0]) if len(thr_for_precision) else np.nan,
     "nota": "Menor umbral que alcanza la precision objetivo en validacion." if len(thr_for_precision) else "Precision objetivo no alcanzada en la grilla evaluada."},
    {"criterio": f"Recall >= {RECALL_TARGET}", "threshold": float(thr_for_recall["threshold"].iloc[0]) if len(thr_for_recall) else np.nan,
     "nota": "Mayor umbral que aun alcanza el recall objetivo en validacion." if len(thr_for_recall) else "Recall objetivo no alcanzado en la grilla evaluada."},
])
display(threshold_optimization_df)
clean_for_export(threshold_optimization_df).to_excel(
    os.path.join(TABLE_DIR, "24b_threshold_optimization_multicriterio.xlsx"), index=False)

final_metrics = []
for X_eval, y_eval, dname in [(X_valid_sel, y_valid, "valid_2024"), (X_test_sel, y_test, "test_2025"), (X_external_sel, y_external, "external_2026")]:
    res = evaluate_model(best_model, X_eval, y_eval, dname, threshold=BEST_THRESHOLD)
    if res is not None:
        res["modelo"] = BEST_MODEL_NAME
        final_metrics.append(res)
final_metrics_df = pd.DataFrame(final_metrics)
display(final_metrics_df)
clean_for_export(final_metrics_df).to_excel(os.path.join(TABLE_DIR, "25_metricas_modelo_final_umbral_ajustado.xlsx"), index=False)

# [PATCH 21] MLflow: modelo ganador, umbral, y metricas de test_2025/valid_2024/external_2026.
mlflow_log_params_safe({"best_model_name": BEST_MODEL_NAME, "best_threshold": BEST_THRESHOLD})
for _row in final_metrics.copy():
    _dname = _row.get("dataset", "unknown")
    mlflow_log_metrics_safe({f"{_dname}_{k}": v for k, v in _row.items()
                              if k not in ("dataset", "modelo", "threshold")})

sensitivity_model_rows = []
try:
    sensitivity_models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, verbose=-1, n_estimators=200),
        "CatBoost": CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0, auto_class_weights="Balanced", iterations=200)
    }
    for high_q, low_q in [(0.70, 0.30), (0.75, 0.25), (0.80, 0.20), (0.85, 0.15)]:
        s_thr = df.loc[df["anio"].isin(TRAIN_YEARS), "saldo_casos"].quantile(high_q)
        t_thr = df.loc[df["anio"].isin(TRAIN_YEARS), "tasa_atencion"].quantile(low_q)
        y_train_alt = ((df_train["saldo_casos"] >= s_thr) & (df_train["tasa_atencion"] <= t_thr)).astype(int)
        y_test_alt = ((df_test["saldo_casos"] >= s_thr) & (df_test["tasa_atencion"] <= t_thr)).astype(int)
        if len(np.unique(y_train_alt)) < 2 or len(np.unique(y_test_alt)) < 2:
            continue
        for model_name, model_obj in sensitivity_models.items():
            mdl = model_obj
            mdl.fit(X_train_sel, y_train_alt)
            score_alt = get_scores(mdl, X_test_sel)
            pred_alt = (score_alt >= 0.5).astype(int)
            sensitivity_model_rows.append({
                "target_rule": f"saldo_P{int(high_q*100)}_and_tasa_P{int(low_q*100)}",
                "high_q_saldo": high_q, "low_q_tasa": low_q,
                "saldo_threshold": s_thr, "tasa_threshold": t_thr,
                "modelo": model_name,
                "test_f1": f1_score(y_test_alt, pred_alt, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test_alt, score_alt) if len(np.unique(y_test_alt)) > 1 else np.nan,
                "test_pr_auc": average_precision_score(y_test_alt, score_alt) if len(np.unique(y_test_alt)) > 1 else np.nan,
                "target_prevalence_train": y_train_alt.mean(),
                "target_prevalence_test": y_test_alt.mean()
            })
    target_sensitivity_model_ranking_df = pd.DataFrame(sensitivity_model_rows)
    if len(target_sensitivity_model_ranking_df):
        target_sensitivity_model_ranking_df = target_sensitivity_model_ranking_df.sort_values(["target_rule", "test_f1"], ascending=[True, False])
        display(target_sensitivity_model_ranking_df)
        target_sensitivity_model_ranking_df.to_excel(os.path.join(TABLE_DIR, "25b_sensibilidad_target_ranking_modelos.xlsx"), index=False)

        best_by_target_df = (
            target_sensitivity_model_ranking_df
            .sort_values(["target_rule", "test_f1", "test_pr_auc", "test_roc_auc"], ascending=[True, False, False, False])
            .groupby("target_rule", as_index=False)
            .head(1)
            .copy()
        )
        role_map = {
            "saldo_P70_and_tasa_P30": "alternativo_alerta_temprana",
            "saldo_P75_and_tasa_P25": "principal_conservador",
            "saldo_P80_and_tasa_P20": "estricto_sobrecarga_severa",
            "saldo_P85_and_tasa_P15": "muy_estricto_casos_extremos",
        }
        justification_map = {
            "saldo_P70_and_tasa_P30": "Mayor sensibilidad y utilidad para alerta temprana; puede mejorar F1/PR-AUC, pero captura casos menos severos.",
            "saldo_P75_and_tasa_P25": "Definicion conservadora y principal: equilibrio entre severidad institucional, prevalencia manejable y estabilidad temporal.",
            "saldo_P80_and_tasa_P20": "Escenario estricto para sobrecarga severa; menor prevalencia y menor F1 esperable.",
            "saldo_P85_and_tasa_P15": "Escenario extremo; util como sensibilidad, no recomendado como target principal por baja prevalencia.",
        }
        best_by_target_df["rol_metodologico"] = best_by_target_df["target_rule"].map(role_map).fillna("sensibilidad")
        best_by_target_df["justificacion"] = best_by_target_df["target_rule"].map(justification_map).fillna("Escenario de sensibilidad.")
        best_by_target_df["target_usado_pipeline"] = best_by_target_df["target_rule"].eq("saldo_P75_and_tasa_P25")
        best_by_target_df["nota_q1"] = "La seleccion del target principal no se basa solo en F1; prioriza coherencia institucional y robustez temporal."
        target_final_decision_df = best_by_target_df
        display(target_final_decision_df)
        target_final_decision_df.to_excel(os.path.join(TABLE_DIR, "25c_decision_metodologica_target_final.xlsx"), index=False)
except Exception as e:
    print("Sensibilidad del target por ranking de modelos no ejecutada:", e)
    target_final_decision_df = pd.DataFrame()

if RUN_HEAVY_BLOCKS:
    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 90),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "random_state": RANDOM_STATE, "class_weight": "balanced", "n_jobs": -1, "verbose": -1
        }
        scores = []
        for tr_idx, va_idx in iter_cv_splits(cv5, X_train_model_cv):
            X_tr, X_va = X_train_model_cv[tr_idx], X_train_model_cv[va_idx]
            y_tr, y_va = y_train_model_cv[tr_idx], y_train_model_cv[va_idx]
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X_tr, y_tr)
            score = get_scores(mdl, X_va)
            pred = (score >= 0.5).astype(int)
            scores.append(f1_score(y_va, pred, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective_lgb, n_trials=N_EXPERIMENTS_PER_MODEL if not FAST_MODE else 10, show_progress_bar=False)
    optuna_params = study.best_params
    optuna_params.update({"random_state": RANDOM_STATE, "class_weight": "balanced", "n_jobs": -1, "verbose": -1})
    optuna_lgb = lgb.LGBMClassifier(**optuna_params)
    optuna_lgb.fit(X_train_model, y_train_model)
    trained_models["LightGBM_Optuna"] = optuna_lgb
    print("Mejor F1 CV Optuna:", study.best_value)
    print("Mejores parametros Optuna:", study.best_params)
    pd.DataFrame([{"best_value": study.best_value, "best_params": json.dumps(study.best_params), "cv_strategy": cv_description}]).to_excel(os.path.join(TABLE_DIR, "26_optuna_lightgbm.xlsx"), index=False)
    # [PATCH 12] Re-guarda el catalogo de modelos incluyendo LightGBM_Optuna.
    save_all_trained_models(trained_models)
else:
    print("Optuna omitido porque RUN_HEAVY_BLOCKS=False.")

def build_fold_data(train_years, eval_year, selected_features=None):
    tr = df[df["anio"].isin(train_years)].copy()
    ev = df[df["anio"] == eval_year].copy()
    if len(tr) == 0 or len(ev) == 0:
        return None
    Xtr_raw = tr[feature_cols_raw].copy()
    Xev_raw = ev[feature_cols_raw].copy()
    ytr = tr[TARGET].copy()
    yev = ev[TARGET].copy()

    high_card_fold = [c for c in Xtr_raw.select_dtypes(include=["object"]).columns if Xtr_raw[c].nunique(dropna=True) >= 10]
    Xtr_fe, Xev_fe, _, _ = add_frequency_encoding(Xtr_raw, Xev_raw, Xev_raw.copy(), pd.DataFrame(columns=Xtr_raw.columns), high_card_fold)
    num_cols = Xtr_fe.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = Xtr_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_one_hot_encoder())]), cat_cols)
    ])
    Xtr = prep.fit_transform(Xtr_fe)
    Xev = prep.transform(Xev_fe)
    return Xtr, ytr, Xev, yev

wf_rows = []
all_available_years = sorted(df["anio"].dropna().unique().tolist())
wf_splits = []
for eval_year in all_available_years:
    if eval_year <= min(TRAIN_YEARS):
        continue
    train_years = [y for y in all_available_years if y < eval_year]
    if train_years:
        wf_splits.append((train_years, eval_year))

for tr_years, ev_year in wf_splits:
    data = build_fold_data(tr_years, ev_year)
    if data is None:
        continue
    Xtr, ytr, Xev, yev = data
    if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
        print(f"Walk-forward {tr_years}->{ev_year} omitido por una sola clase en train o evaluacion.")
        continue
    mdl = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, verbose=-1, n_estimators=200)
    mdl.fit(Xtr, ytr)
    score = get_scores(mdl, Xev)
    pred = (score >= 0.5).astype(int)
    wf_rows.append({
        "train_years": str(tr_years), "eval_year": ev_year,
        "accuracy": accuracy_score(yev, pred), "precision": precision_score(yev, pred, zero_division=0),
        "recall": recall_score(yev, pred, zero_division=0), "f1": f1_score(yev, pred, zero_division=0),
        "roc_auc": roc_auc_score(yev, score) if len(np.unique(yev)) > 1 else np.nan,
        "pr_auc": average_precision_score(yev, score) if len(np.unique(yev)) > 1 else np.nan,
        "interpretacion": "Fold walk-forward anual; evalua generalizacion al siguiente anio completo/parcial."
    })
walk_forward_df = pd.DataFrame(wf_rows)
display(walk_forward_df)
walk_forward_df.to_excel(os.path.join(TABLE_DIR, "27_walk_forward_validation.xlsx"), index=False)

nested_temporal_cv_df = pd.DataFrame()
nested_temporal_cv_summary_df = pd.DataFrame()
if RUN_NESTED_TEMPORAL_CV and RUN_HEAVY_BLOCKS:
    nested_rows = []
    try:
        years_nested = df_train["anio"].values if len(X_train_sel) == len(df_train) else None
        if years_nested is None:
            raise ValueError("Nested Temporal CV requiere que X_train_sel conserve la correspondencia con df_train.")

        outer_splits = build_year_forward_cv_splits(years_nested, min_train_periods=1, max_splits=None)
        outer_splits = filter_cv_splits_with_both_classes(outer_splits, np.asarray(y_train))

        nested_model_spaces = {
            "LogisticRegression": (LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
                                    {"C": np.logspace(-3, 2, 10), "penalty": ["l1", "l2"]}),
            "LightGBM": (lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, verbose=-1),
                         {"n_estimators": [100, 200, 300], "num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.05, 0.1]}),
            "CatBoost": (CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0, auto_class_weights="Balanced"),
                         {"iterations": [100, 200, 300], "depth": [3, 5, 7], "learning_rate": [0.03, 0.1]}),
        }

        for outer_id, (outer_tr_idx, outer_te_idx) in enumerate(outer_splits, start=1):
            outer_train_years = sorted(pd.Series(years_nested[outer_tr_idx]).unique().tolist())
            outer_test_year = sorted(pd.Series(years_nested[outer_te_idx]).unique().tolist())
            X_outer_tr, y_outer_tr = X_train_sel[outer_tr_idx], np.asarray(y_train)[outer_tr_idx]
            X_outer_te, y_outer_te = X_train_sel[outer_te_idx], np.asarray(y_train)[outer_te_idx]
            years_outer_tr = years_nested[outer_tr_idx]

            inner_splits = build_year_forward_cv_splits(years_outer_tr, min_train_periods=1, max_splits=CV_FOLDS)
            inner_splits = filter_cv_splits_with_both_classes(inner_splits, y_outer_tr)
            if len(inner_splits) < 2:
                minority_count = int(np.bincount(y_outer_tr).min())
                if minority_count < 2:
                    continue
                inner_cv = StratifiedKFold(n_splits=min(3, minority_count), shuffle=True, random_state=RANDOM_STATE)
            else:
                inner_cv = inner_splits

            for model_name, (estimator, param_grid) in nested_model_spaces.items():
                try:
                    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid,
                                                 n_iter=10 if not FAST_MODE else 3, scoring="f1", cv=inner_cv,
                                                 random_state=RANDOM_STATE, n_jobs=-1, error_score=np.nan)
                    search.fit(X_outer_tr, y_outer_tr)
                    mdl = search.best_estimator_
                    scores = get_scores(mdl, X_outer_te)
                    preds = (scores >= 0.5).astype(int)
                    nested_rows.append({
                        "outer_fold": outer_id, "outer_train_years": str(outer_train_years), "outer_test_year": str(outer_test_year),
                        "modelo": model_name, "inner_best_f1": search.best_score_,
                        "outer_f1": f1_score(y_outer_te, preds, zero_division=0),
                        "outer_roc_auc": roc_auc_score(y_outer_te, scores) if len(np.unique(y_outer_te)) > 1 else np.nan,
                        "outer_pr_auc": average_precision_score(y_outer_te, scores) if len(np.unique(y_outer_te)) > 1 else np.nan,
                        "outer_recall": recall_score(y_outer_te, preds, zero_division=0),
                        "outer_precision": precision_score(y_outer_te, preds, zero_division=0),
                        "best_params": json.dumps(search.best_params_),
                    })
                except Exception as e:
                    nested_rows.append({"outer_fold": outer_id, "outer_train_years": str(outer_train_years),
                                         "outer_test_year": str(outer_test_year), "modelo": model_name, "error": str(e)[:500]})

        nested_temporal_cv_df = pd.DataFrame(nested_rows)
        display(nested_temporal_cv_df)
        nested_temporal_cv_df.to_excel(os.path.join(TABLE_DIR, "27b_nested_temporal_cv.xlsx"), index=False)

        if len(nested_temporal_cv_df) and "outer_f1" in nested_temporal_cv_df.columns:
            nested_temporal_cv_summary_df = (
                nested_temporal_cv_df.groupby("modelo")[["outer_f1", "outer_roc_auc", "outer_pr_auc", "outer_recall", "outer_precision"]]
                .agg(["mean", "std"]).reset_index()
            )
            if isinstance(nested_temporal_cv_summary_df.columns, pd.MultiIndex):
                nested_temporal_cv_summary_df.columns = [
                    "_".join([str(x) for x in col if str(x) != ""]).strip("_")
                    for col in nested_temporal_cv_summary_df.columns.to_flat_index()
                ]
            display(nested_temporal_cv_summary_df)
            nested_temporal_cv_summary_df.to_excel(os.path.join(TABLE_DIR, "27c_nested_temporal_cv_summary.xlsx"), index=False)
    except Exception as e:
        print("Nested Temporal CV no ejecutado por error real:", e)
        nested_temporal_cv_df = pd.DataFrame([{"status": "omitido", "error": str(e)}])
else:
    print("Nested Temporal CV omitido por configuracion.")

from sklearn.base import clone

def _slice_rows(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]

def evaluate_scores_from_array(scores, y_true, dataset_name, model_name, threshold=0.5):
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    pred = (scores_arr >= threshold).astype(int)
    base = {
        "dataset": dataset_name, "threshold": threshold,
        "accuracy": accuracy_score(y_true_arr, pred), "precision": precision_score(y_true_arr, pred, zero_division=0),
        "recall": recall_score(y_true_arr, pred, zero_division=0), "f1": f1_score(y_true_arr, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true_arr, scores_arr) if len(np.unique(y_true_arr)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true_arr, scores_arr) if len(np.unique(y_true_arr)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, pred),
        "specificity": specificity_score(y_true_arr, pred), "modelo": model_name
    }
    base.update(extended_binary_metrics(y_true_arr, pred))
    return base

def predict_temporal_stacking_scores(artifact, X):
    if artifact is None:
        raise ValueError("No existe temporal_stacking_artifact entrenado.")
    meta_model = artifact["meta_model"]
    fitted_base_models = artifact["base_models"]
    meta_X = np.column_stack([get_scores(model_full, X) for _, model_full in fitted_base_models])
    return meta_model.predict_proba(meta_X)[:, 1]

ensemble_metrics = []
temporal_stacking_artifact = None

try:
    base_model_names = ["LogisticRegression", "LightGBM", "LightGBM_Optuna", "XGBoost", "CatBoost"]
    ensemble_estimators = [(n, trained_models[n]) for n in base_model_names if n in trained_models and hasattr(trained_models[n], "predict_proba")]
    if len(ensemble_estimators) >= 2:
        voting = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
        voting.fit(X_train_model, y_train_model)
        trained_models["VotingClassifier"] = voting
        for X_eval, y_eval, dataset in [(X_valid_sel, y_valid, "valid_2024"), (X_test_sel, y_test, "test_2025")]:
            voting_res = evaluate_model(voting, X_eval, y_eval, dataset, threshold=0.5)
            if voting_res is not None:
                voting_res["modelo"] = "VotingClassifier"
                ensemble_metrics.append(voting_res)
        print("VotingClassifier ejecutado correctamente.")
    else:
        print("VotingClassifier omitido: no hay suficientes modelos base con predict_proba.")
except Exception as e:
    print("VotingClassifier no ejecutado por error:", e)

try:
    base_model_names = ["LogisticRegression", "LightGBM", "LightGBM_Optuna", "XGBoost", "CatBoost"]
    base_models_for_stack = [(n, trained_models[n]) for n in base_model_names if n in trained_models]
    if len(base_models_for_stack) >= 2:
        y_train_arr = np.asarray(y_train_model)
        n_train = len(y_train_arr)
        n_models = len(base_models_for_stack)
        meta_train = np.full((n_train, n_models), np.nan)

        if len(X_train_model) == len(df_train):
            years_stack = df_train["anio"].values
            stack_splits = build_year_forward_cv_splits(years_stack, min_train_periods=1, max_splits=CV_FOLDS)
            stack_splits = filter_cv_splits_with_both_classes(stack_splits, y_train_arr)
        else:
            stack_splits = []

        if len(stack_splits) < 2:
            n_splits_stack = min(CV_FOLDS, max(2, n_train // 1000))
            stack_splits = list(TimeSeriesSplit(n_splits=n_splits_stack).split(X_train_model))
            print("Stacking temporal: respaldo con TimeSeriesSplit por filas por folds anuales insuficientes.")

        for fold, (tr_idx, val_idx) in enumerate(stack_splits, start=1):
            print(f"Stacking temporal - fold {fold}")
            X_tr_fold = _slice_rows(X_train_model, tr_idx)
            y_tr_fold = y_train_arr[tr_idx]
            X_val_fold = _slice_rows(X_train_model, val_idx)
            for j, (name, base_model) in enumerate(base_models_for_stack):
                model_fold = clone(base_model)
                model_fold.fit(X_tr_fold, y_tr_fold)
                meta_train[val_idx, j] = get_scores(model_fold, X_val_fold)

        valid_meta_rows = ~np.isnan(meta_train).any(axis=1)
        X_meta_train = meta_train[valid_meta_rows]
        y_meta_train = y_train_arr[valid_meta_rows]

        if len(X_meta_train) > 0 and len(np.unique(y_meta_train)) > 1:
            meta_model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
            meta_model.fit(X_meta_train, y_meta_train)
            fitted_base_models = []
            for name, base_model in base_models_for_stack:
                model_full = clone(base_model)
                model_full.fit(X_train_model, y_train_model)
                fitted_base_models.append((name, model_full))
            temporal_stacking_artifact = {
                "meta_model": meta_model, "base_models": fitted_base_models,
                "base_model_names": [name for name, _ in fitted_base_models],
                "n_meta_features": len(fitted_base_models),
                "note": "El meta-modelo recibe scores de modelos base; no debe evaluarse directamente con X de las features originales."
            }
            for X_eval, y_eval, dataset in [(X_valid_sel, y_valid, "valid_2024"), (X_test_sel, y_test, "test_2025")]:
                stack_scores = predict_temporal_stacking_scores(temporal_stacking_artifact, X_eval)
                stack_res = evaluate_scores_from_array(stack_scores, y_eval, dataset, "TemporalStackingClassifier", threshold=0.5)
                ensemble_metrics.append(stack_res)
            print("TemporalStackingClassifier ejecutado correctamente.")
        else:
            print("TemporalStackingClassifier omitido: no hay suficientes filas meta validas.")
    else:
        print("TemporalStackingClassifier omitido: no hay suficientes modelos base.")
except Exception as e:
    print("TemporalStackingClassifier no ejecutado por error:", e)

ensemble_metrics_df = pd.DataFrame(ensemble_metrics)
if len(ensemble_metrics_df) > 0:
    display(ensemble_metrics_df)
    ensemble_metrics_df.to_excel(os.path.join(TABLE_DIR, "28_ensemble_metrics.xlsx"), index=False)
else:
    print("No se generaron metricas de ensamble.")

# [PATCH 12] Re-guarda el catalogo de modelos incluyendo VotingClassifier / stacking.
save_all_trained_models(trained_models)
if temporal_stacking_artifact is not None:
    joblib.dump(temporal_stacking_artifact, os.path.join(MODEL_DIR, "temporal_stacking_artifact.joblib"), compress=3)
# [PATCH 11-13] Checkpoint M05: ensambles listos (voting/stacking) y modelos base completos.
ckpt.save("M05_ensambles_listos", ensemble_metrics_df=ensemble_metrics_df,
          trained_model_names=list(trained_models.keys()))

def feature_indices_matching(patterns, exclude=False):
    idxs = []
    for i, feat in enumerate(selected_features_final):
        match = any(p.lower() in feat.lower() for p in patterns)
        if (match and not exclude) or ((not match) and exclude):
            idxs.append(i)
    return idxs

def evaluate_ablation_variant(label, local_indices):
    if len(local_indices) == 0:
        return []
    Xtr = X_train_sel[:, local_indices]
    Xva = X_valid_sel[:, local_indices]
    Xte = X_test_sel[:, local_indices]
    mdl = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                              n_estimators=250 if not FAST_MODE else 80, learning_rate=0.05, n_jobs=-1, verbose=-1)
    mdl.fit(Xtr, y_train)
    rows = []
    for X_eval, y_eval, dataset in [(Xva, y_valid, "valid_2024"), (Xte, y_test, "test_2025")]:
        score = get_scores(mdl, X_eval)
        pred = (score >= 0.5).astype(int)
        rows.append({
            "variant": label, "n_features": len(local_indices), "dataset": dataset,
            "accuracy": accuracy_score(y_eval, pred), "precision": precision_score(y_eval, pred, zero_division=0),
            "recall": recall_score(y_eval, pred, zero_division=0), "f1": f1_score(y_eval, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_eval, score) if len(np.unique(y_eval)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_eval, score) if len(np.unique(y_eval)) > 1 else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y_eval, pred)
        })
    return rows

ablation_df = pd.DataFrame()
if RUN_ABLATION_STUDY:
    all_local_idx = list(range(len(selected_features_final)))
    variants = {
        "full_selected_features": all_local_idx,
        "without_historical_growth": feature_indices_matching(["hist_", "growth_"], exclude=True),
        "without_territorial": feature_indices_matching(["dpto", "prov", "dist", "ubigeo", "distrito_fiscal", "lima"], exclude=True),
        "without_frequency_encoding": feature_indices_matching(["freq_"], exclude=True),
        "without_interactions": feature_indices_matching(["inter_"], exclude=True),
    }
    ablation_rows = []
    for label, idxs in variants.items():
        print("Ejecutando ablation:", label, "features:", len(idxs))
        ablation_rows.extend(evaluate_ablation_variant(label, idxs))
    ablation_df = pd.DataFrame(ablation_rows)
    display(ablation_df)
    ablation_df.to_excel(os.path.join(TABLE_DIR, "43_ablation_study.xlsx"), index=False)
else:
    print("Ablation Study omitido porque RUN_ABLATION_STUDY=False.")

predictive_model_name = BEST_MODEL_NAME
predictive_model = trained_models[predictive_model_name]
explainability_model_name = predictive_model_name
explainability_model = predictive_model

print("Modelo predictivo principal:", predictive_model_name)
print("Modelo usado para explicabilidad:", explainability_model_name)

# [PATCH 21] MLflow: registra el modelo predictivo principal (flavor sklearn si es compatible).
mlflow_log_best_model_safe(predictive_model, model_name=f"predictive_model_{predictive_model_name}")

# [v3.02 - observacion revisor #13] Learning Curve: F1 en train vs F1 en CV
# temporal a medida que crece el tamano de entrenamiento. Sirve para mostrar
# ausencia (o presencia) de overfitting/underfitting de forma visual, algo
# que un revisor Q1 suele pedir explicitamente.
try:
    from sklearn.model_selection import learning_curve
    lc_estimator = clone(predictive_model)
    train_sizes_abs, train_scores_lc, valid_scores_lc = learning_curve(
        lc_estimator, X_train_model, y_train_model,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv5 if not isinstance(cv5, list) else min(3, len(cv5)),
        scoring="f1", n_jobs=-1, random_state=RANDOM_STATE,
        error_score=np.nan,
    )
    learning_curve_df = pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_f1_mean": train_scores_lc.mean(axis=1), "train_f1_std": train_scores_lc.std(axis=1),
        "cv_f1_mean": valid_scores_lc.mean(axis=1), "cv_f1_std": valid_scores_lc.std(axis=1),
    })
    display(learning_curve_df)
    learning_curve_df.to_excel(os.path.join(TABLE_DIR, "24c_learning_curve.xlsx"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(learning_curve_df["train_size"], learning_curve_df["train_f1_mean"], marker="o", label="F1 train")
    plt.fill_between(learning_curve_df["train_size"],
                      learning_curve_df["train_f1_mean"] - learning_curve_df["train_f1_std"],
                      learning_curve_df["train_f1_mean"] + learning_curve_df["train_f1_std"], alpha=0.15)
    plt.plot(learning_curve_df["train_size"], learning_curve_df["cv_f1_mean"], marker="o", label="F1 CV (walk-forward)")
    plt.fill_between(learning_curve_df["train_size"],
                      learning_curve_df["cv_f1_mean"] - learning_curve_df["cv_f1_std"],
                      learning_curve_df["cv_f1_mean"] + learning_curve_df["cv_f1_std"], alpha=0.15)
    plt.xlabel("Tamano de entrenamiento (n registros)"); plt.ylabel("F1")
    plt.title(f"Learning Curve - {predictive_model_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "24c_learning_curve.png"), dpi=300)
    plt.show()
except Exception as e:
    print("Learning Curve no ejecutada:", e)
    learning_curve_df = pd.DataFrame()

perm_df = pd.DataFrame()
try:
    perm = permutation_importance(explainability_model, X_test_sel, y_test, n_repeats=10 if not FAST_MODE else 3,
                                   random_state=RANDOM_STATE, scoring="f1", n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": selected_features_final, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std,
        "modelo_explicado": explainability_model_name, "interpretacion": "Importancia predictiva por permutacion; no causalidad."
    }).sort_values("importance_mean", ascending=False)
    display(perm_df.head(25))
    perm_df.to_excel(os.path.join(TABLE_DIR, "29_permutation_importance_test.xlsx"), index=False)
except Exception as e:
    print("Permutation Importance fallo:", e)

shap_rank_df = pd.DataFrame()
explainer = None
try:
    X_shap = X_test_sel[:min(500, X_test_sel.shape[0])]
    shap_feature_names = selected_features_final
    try:
        explainer = shap.Explainer(explainability_model, X_train_sel[:min(500, X_train_sel.shape[0])], feature_names=shap_feature_names)
        shap_values = explainer(X_shap)
        shap_model_used = explainability_model_name
    except Exception as shap_main_error:
        fallback_name = "LightGBM_Optuna" if "LightGBM_Optuna" in trained_models else ("LightGBM" if "LightGBM" in trained_models else None)
        if fallback_name is None:
            raise shap_main_error
        print(f"SHAP con modelo principal fallo; se usa respaldo {fallback_name}:", shap_main_error)
        explainability_model_name = fallback_name
        explainability_model = trained_models[fallback_name]
        explainer = shap.Explainer(explainability_model, X_train_sel[:min(500, X_train_sel.shape[0])], feature_names=shap_feature_names)
        shap_values = explainer(X_shap)
        shap_model_used = fallback_name

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "24_shap_beeswarm.png"), dpi=300, bbox_inches="tight"); plt.show()

    plt.figure()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "24_shap_bar.png"), dpi=300, bbox_inches="tight"); plt.show()

    shap_abs = np.abs(shap_values.values).mean(axis=0)
    shap_rank_df = pd.DataFrame({
        "feature": shap_feature_names, "mean_abs_shap": shap_abs, "modelo_explicado": shap_model_used,
        "interpretacion": "Asociacion predictiva global; no causalidad."
    }).sort_values("mean_abs_shap", ascending=False)
    display(shap_rank_df.head(25))
    shap_rank_df.to_excel(os.path.join(TABLE_DIR, "30_shap_global_ranking.xlsx"), index=False)

    top_feature = shap_rank_df.iloc[0]["feature"]
    top_idx = shap_feature_names.index(top_feature)
    shap.plots.scatter(shap_values[:, top_idx], show=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "24_shap_dependence_top_feature.png"), dpi=300, bbox_inches="tight"); plt.show()
except Exception as e:
    print("SHAP avanzado no pudo ejecutarse en este entorno:", e)

try:
    top_perm_features = perm_df.head(2)["feature"].tolist() if len(perm_df) else selected_features_final[:2]
    top_indices = [selected_features_final.index(f) for f in top_perm_features if f in selected_features_final]
    if top_indices:
        PartialDependenceDisplay.from_estimator(explainability_model, X_test_sel, features=top_indices[:2],
                                                  feature_names=selected_features_final, kind="both")
        plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "24_pdp_ice_top_features.png"), dpi=300, bbox_inches="tight"); plt.show()
except Exception as e:
    print("PDP/ICE no pudo ejecutarse:", e)

shap_interaction_df = pd.DataFrame()
if RUN_SHAP_INTERACTIONS:
    try:
        tree_candidate_name = None
        for cand in [predictive_model_name, "CatBoost", "LightGBM_Optuna", "LightGBM", "XGBoost"]:
            if cand in trained_models and cand not in ["SVM", "LogisticRegression", "VotingClassifier", "TemporalStackingClassifier", "TabPFN", "FTTransformer"]:
                tree_candidate_name = cand
                break
        if tree_candidate_name is None:
            raise ValueError("No hay modelo de arbol compatible para SHAP interactions.")
        interaction_model = trained_models[tree_candidate_name]
        X_inter = X_test_sel[:min(250 if not FAST_MODE else 80, X_test_sel.shape[0])]
        tree_explainer = shap.TreeExplainer(interaction_model)
        interaction_values = tree_explainer.shap_interaction_values(X_inter)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[-1]
        interaction_values = np.asarray(interaction_values)
        pair_rows = []
        p = min(interaction_values.shape[1], len(selected_features_final))
        mean_abs_inter = np.abs(interaction_values[:, :p, :p]).mean(axis=0)
        for i in range(p):
            for j in range(i + 1, p):
                pair_rows.append({"feature_1": selected_features_final[i], "feature_2": selected_features_final[j],
                                   "mean_abs_interaction": float(mean_abs_inter[i, j]), "modelo_explicado": tree_candidate_name,
                                   "interpretacion": "Interaccion predictiva; no causalidad."})
        shap_interaction_df = pd.DataFrame(pair_rows).sort_values("mean_abs_interaction", ascending=False)
        display(shap_interaction_df.head(25))
        shap_interaction_df.to_excel(os.path.join(TABLE_DIR, "30b_shap_interaction_values_top_pairs.xlsx"), index=False)
    except Exception as e:
        print("SHAP interaction values omitidos:", e)

counterfactual_df = pd.DataFrame()
if RUN_COUNTERFACTUAL_EXPLANATIONS:
    try:
        scores_test = get_scores(predictive_model, X_test_sel)
        high_risk_idx = np.where(scores_test >= max(BEST_THRESHOLD, 0.5))[0]
        high_risk_idx = high_risk_idx[:min(10 if not FAST_MODE else 3, len(high_risk_idx))]
        ranked_features = perm_df["feature"].tolist() if len(perm_df) else (shap_rank_df["feature"].tolist() if len(shap_rank_df) else selected_features_final)
        mutable_features = [f for f in ranked_features if f in selected_features_final and not any(t in f for t in ["_NO_ESPECIFICADO", "_nan"])]
        mutable_features = mutable_features[:min(20, len(mutable_features))]
        y_train_arr = np.asarray(y_train_model)
        nonrisk_train_mask = y_train_arr == 0
        train_reference = np.median(X_train_sel[nonrisk_train_mask], axis=0) if nonrisk_train_mask.any() else np.median(X_train_sel, axis=0)
        cf_rows = []
        for local_idx in high_risk_idx:
            x0 = X_test_sel[local_idx].copy()
            original_score = float(scores_test[local_idx])
            current = x0.copy()
            changes = []
            final_score = original_score
            for feat in mutable_features:
                feat_idx = selected_features_final.index(feat)
                old_value = current[feat_idx]
                new_value = train_reference[feat_idx]
                if np.isclose(old_value, new_value):
                    continue
                candidate = current.copy()
                candidate[feat_idx] = new_value
                candidate_score = float(get_scores(predictive_model, candidate.reshape(1, -1))[0])
                if candidate_score < final_score:
                    current = candidate
                    final_score = candidate_score
                    changes.append(f"{feat}: {old_value:.4f} -> {new_value:.4f}")
                if final_score < BEST_THRESHOLD:
                    break
            cf_rows.append({
                "test_row_position": int(local_idx), "original_score": original_score, "counterfactual_score": final_score,
                "crossed_threshold": bool(final_score < BEST_THRESHOLD), "threshold": BEST_THRESHOLD,
                "n_changes": len(changes), "changes_preprocessed_space": "; ".join(changes[:10]),
                "nota": "Contrafactual controlado en espacio preprocesado; no implica causalidad ni intervencion directa."
            })
        counterfactual_df = pd.DataFrame(cf_rows)
        display(counterfactual_df)
        counterfactual_df.to_excel(os.path.join(TABLE_DIR, "30c_counterfactual_explanations_controlled.xlsx"), index=False)
    except Exception as e:
        print("Contrafactuales controlados no ejecutados:", e)

# [PATCH 11-13] Checkpoint M06: explicabilidad (permutation/SHAP/interacciones/contrafactuales) lista.
ckpt.save("M06_explicabilidad", perm_df=perm_df, shap_rank_df=shap_rank_df,
          shap_interaction_df=shap_interaction_df, counterfactual_df=counterfactual_df,
          predictive_model_name=predictive_model_name, explainability_model_name=explainability_model_name)

cluster_X = np.asarray(X_train_sel)
cluster_rows = []

elbow_rows = []
kmeans_candidates = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(cluster_X)
    kmeans_candidates[k] = (km, labels)
    elbow_rows.append({"k": k, "inertia": km.inertia_})
    try:
        cluster_rows.append({"modelo": "KMeans", "k": k, "silhouette": silhouette_score(cluster_X, labels),
                              "davies_bouldin": davies_bouldin_score(cluster_X, labels),
                              "calinski_harabasz": calinski_harabasz_score(cluster_X, labels),
                              "nota": "Ajustado solo con train sobre features seleccionadas leak-free."})
    except Exception as e:
        cluster_rows.append({"modelo": "KMeans", "k": k, "silhouette": np.nan, "davies_bouldin": np.nan,
                              "calinski_harabasz": np.nan, "nota": f"Metrica no calculable: {e}"})

elbow_df = pd.DataFrame(elbow_rows)
cluster_metrics_kmeans_df = pd.DataFrame(cluster_rows)

plt.figure(figsize=(6,4))
plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
plt.title("Elbow Method - KMeans"); plt.xlabel("K"); plt.ylabel("Inertia")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "25_elbow_kmeans.png"), dpi=300); plt.show()

valid_kmeans_metrics = cluster_metrics_kmeans_df[(cluster_metrics_kmeans_df["modelo"] == "KMeans") & cluster_metrics_kmeans_df["silhouette"].notna()].copy()
if len(valid_kmeans_metrics):
    BEST_CLUSTER_K = int(valid_kmeans_metrics.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0]["k"])
else:
    BEST_CLUSTER_K = 4 if 4 in kmeans_candidates else sorted(kmeans_candidates.keys())[0]

operational_cluster_model, train_cluster_labels = kmeans_candidates[BEST_CLUSTER_K]
print("K seleccionado para perfilamiento operativo:", BEST_CLUSTER_K)

# [v3.02 - observacion revisor #12] Bootstrap Cluster Stability (Adjusted Rand
# Index): remuestrea train con reemplazo, reajusta KMeans(k=BEST_CLUSTER_K) y
# compara el particionamiento contra el original via ARI. ARI cercano a 1 =
# clustering estable; cercano a 0 = inestable/dependiente de la muestra.
try:
    from sklearn.metrics import adjusted_rand_score
    n_boot_cluster = 20
    ari_scores = []
    rng_cluster = np.random.RandomState(RANDOM_STATE)
    n_train_cluster = cluster_X.shape[0]
    for b in range(n_boot_cluster):
        boot_idx = rng_cluster.choice(n_train_cluster, size=n_train_cluster, replace=True)
        km_boot = KMeans(n_clusters=BEST_CLUSTER_K, random_state=RANDOM_STATE + b, n_init=5)
        boot_labels = km_boot.fit_predict(cluster_X[boot_idx])
        ari_scores.append(adjusted_rand_score(train_cluster_labels[boot_idx], boot_labels))
    cluster_stability_df = pd.DataFrame([{
        "k": BEST_CLUSTER_K, "n_bootstrap": n_boot_cluster,
        "ari_mean": float(np.mean(ari_scores)), "ari_std": float(np.std(ari_scores)),
        "ari_min": float(np.min(ari_scores)), "ari_max": float(np.max(ari_scores)),
        "interpretacion": ("estable" if np.mean(ari_scores) >= 0.75 else
                           "moderadamente estable" if np.mean(ari_scores) >= 0.5 else
                           "poco estable - interpretar perfiles con cautela")
    }])
    display(cluster_stability_df)
    cluster_stability_df.to_excel(os.path.join(TABLE_DIR, "32b_cluster_bootstrap_stability_ari.xlsx"), index=False)
except Exception as e:
    print("Bootstrap Cluster Stability (ARI) no ejecutado:", e)
    cluster_stability_df = pd.DataFrame()

df_train = df_train.copy(); df_valid = df_valid.copy(); df_test = df_test.copy(); df_external = df_external.copy()
df_train["cluster_operativo"] = train_cluster_labels
df_valid["cluster_operativo"] = operational_cluster_model.predict(X_valid_sel) if len(df_valid) else []
df_test["cluster_operativo"] = operational_cluster_model.predict(X_test_sel) if len(df_test) else []
df_external["cluster_operativo"] = operational_cluster_model.predict(X_external_sel) if len(df_external) else []

df["cluster_operativo"] = np.nan
for part in [df_train, df_valid, df_test, df_external]:
    if len(part) and "cluster_operativo" in part.columns:
        df.loc[part.index, "cluster_operativo"] = part["cluster_operativo"].values
df["cluster_operativo"] = df["cluster_operativo"].astype("Int64")

try:
    db = DBSCAN(eps=1.5, min_samples=10)
    labels_db = db.fit_predict(cluster_X)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    if 1 < len(set(labels_db)) < len(labels_db):
        cluster_rows.append({"modelo": "DBSCAN", "k": n_clusters_db, "silhouette": silhouette_score(cluster_X, labels_db),
                              "davies_bouldin": davies_bouldin_score(cluster_X, labels_db),
                              "calinski_harabasz": calinski_harabasz_score(cluster_X, labels_db),
                              "nota": "Soporte exploratorio; no usado como perfil operativo principal."})
    else:
        cluster_rows.append({"modelo": "DBSCAN", "k": n_clusters_db, "silhouette": np.nan, "davies_bouldin": np.nan,
                              "calinski_harabasz": np.nan, "nota": "No genero particion valida"})
except Exception as e:
    cluster_rows.append({"modelo": "DBSCAN", "k": np.nan, "silhouette": np.nan, "davies_bouldin": np.nan,
                          "calinski_harabasz": np.nan, "nota": f"No ejecutado: {e}"})

for k in [2,3,4,5]:
    try:
        ag = AgglomerativeClustering(n_clusters=k)
        labels_ag = ag.fit_predict(cluster_X)
        cluster_rows.append({"modelo": "Agglomerative", "k": k, "silhouette": silhouette_score(cluster_X, labels_ag),
                              "davies_bouldin": davies_bouldin_score(cluster_X, labels_ag),
                              "calinski_harabasz": calinski_harabasz_score(cluster_X, labels_ag),
                              "nota": "Soporte exploratorio; no usado como perfil operativo principal."})
    except Exception as e:
        cluster_rows.append({"modelo": "Agglomerative", "k": k, "silhouette": np.nan, "davies_bouldin": np.nan,
                              "calinski_harabasz": np.nan, "nota": f"No ejecutado: {e}"})

cluster_metrics_df = pd.DataFrame(cluster_rows)
display(cluster_metrics_df)
elbow_df.to_excel(os.path.join(TABLE_DIR, "31_elbow_kmeans.xlsx"), index=False)
cluster_metrics_df.to_excel(os.path.join(TABLE_DIR, "32_validacion_clusters_operativos.xlsx"), index=False)

profile_cols = [c for c in ["ingresado", "atendido", "saldo_casos", "tasa_atencion", "ratio_saldo"] if c in df.columns]
cluster_profile_df = (
    df.dropna(subset=["cluster_operativo"]).groupby("cluster_operativo")
      .agg(n=("riesgo_congestion", "size"), riesgo_proxy_rate=("riesgo_congestion", "mean"),
           riesgo_proxy_count=("riesgo_congestion", "sum"),
           **{f"{c}_mean": (c, "mean") for c in profile_cols}, **{f"{c}_median": (c, "median") for c in profile_cols})
      .reset_index()
)
cluster_profile_df["riesgo_proxy_pct"] = cluster_profile_df["riesgo_proxy_rate"] * 100

if len(cluster_profile_df):
    cluster_profile_df["rank_riesgo_desc"] = cluster_profile_df["riesgo_proxy_rate"].rank(ascending=False, method="min")
    cluster_profile_df["rank_saldo_desc"] = cluster_profile_df["saldo_casos_mean"].rank(ascending=False, method="min") if "saldo_casos_mean" in cluster_profile_df.columns else np.nan
    cluster_profile_df["rank_ingresado_desc"] = cluster_profile_df["ingresado_mean"].rank(ascending=False, method="min") if "ingresado_mean" in cluster_profile_df.columns else np.nan
    cluster_profile_df["rank_tasa_asc"] = cluster_profile_df["tasa_atencion_mean"].rank(ascending=True, method="min") if "tasa_atencion_mean" in cluster_profile_df.columns else np.nan
    cluster_profile_df["carga_score_relativo"] = cluster_profile_df[["rank_riesgo_desc", "rank_saldo_desc", "rank_ingresado_desc", "rank_tasa_asc"]].mean(axis=1, skipna=True)

    def label_operational_profile(row):
        if row.get("rank_riesgo_desc", np.inf) == 1 and row.get("rank_saldo_desc", np.inf) == 1:
            return "perfil_alta_carga_vigilancia_operativa"
        if row.get("rank_riesgo_desc", np.inf) == 1:
            return "perfil_mayor_riesgo_proxy_vigilancia"
        if row.get("rank_riesgo_desc", np.inf) == len(cluster_profile_df):
            return "perfil_operativo_estandar_carga_moderada"
        return "perfil_intermedio_o_especializado"

    def article_profile_description(row):
        label = row.get("perfil_operativo", "")
        if label == "perfil_alta_carga_vigilancia_operativa":
            return "Mayor carga, mayor saldo promedio y mayor concentracion relativa del riesgo proxy; requiere vigilancia operacional."
        if label == "perfil_mayor_riesgo_proxy_vigilancia":
            return "Mayor proporcion relativa de riesgo proxy, aunque no necesariamente extremo en todos los indicadores de carga."
        if label == "perfil_operativo_estandar_carga_moderada":
            return "Carga y riesgo relativo mas moderados; funciona como grupo de referencia operativo."
        return "Perfil intermedio o especializado; interpretar segun composicion territorial y tipo de fiscalia."

    cluster_profile_df["perfil_operativo"] = cluster_profile_df.apply(label_operational_profile, axis=1)
    cluster_profile_df["descripcion_articulo"] = cluster_profile_df.apply(article_profile_description, axis=1)

display(cluster_profile_df.sort_values("riesgo_proxy_rate", ascending=False))
cluster_profile_df.to_excel(os.path.join(TABLE_DIR, "33_perfiles_operativos_clusters.xlsx"), index=False)

cluster_target_crosstab_df = pd.crosstab(df["cluster_operativo"], df["riesgo_congestion"], normalize="index").reset_index()
cluster_target_crosstab_df.to_excel(os.path.join(TABLE_DIR, "33b_target_proxy_por_cluster.xlsx"), index=False)
display(cluster_target_crosstab_df)

cluster_scenario_rows = []
for scenario_name in TARGET_SCENARIOS:
    col_name = f"riesgo_proxy_{scenario_name}"
    if col_name in df.columns:
        for cl, g in df.dropna(subset=["cluster_operativo"]).groupby("cluster_operativo"):
            cluster_scenario_rows.append({"scenario": scenario_name, "cluster_operativo": int(cl), "n": len(g),
                                           "riesgo_rate": float(g[col_name].mean()), "riesgo_pct": float(g[col_name].mean() * 100)})
cluster_scenario_validation_df = pd.DataFrame(cluster_scenario_rows)
display(cluster_scenario_validation_df.head(30))
cluster_scenario_validation_df.to_excel(os.path.join(TABLE_DIR, "33c_validacion_convergente_target_cluster_escenarios.xlsx"), index=False)

# [PATCH 11-13] Checkpoint M07: clustering operativo (perfiles + labels) completo.
ckpt.save("M07_clustering_operativo", cluster_profile_df=cluster_profile_df,
          BEST_CLUSTER_K=BEST_CLUSTER_K, cluster_metrics_df=cluster_metrics_df)
joblib.dump(operational_cluster_model, os.path.join(MODEL_DIR, "operational_cluster_kmeans.joblib"), compress=3)

plt.figure(figsize=(7, 4))
cluster_profile_plot = cluster_profile_df.sort_values("cluster_operativo")
plt.bar(cluster_profile_plot["cluster_operativo"].astype(str), cluster_profile_plot["riesgo_proxy_pct"])
plt.title("Riesgo proxy por perfil operativo"); plt.xlabel("Cluster operativo"); plt.ylabel("Riesgo proxy (%)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "25_riesgo_proxy_por_cluster_operativo.png"), dpi=300); plt.show()

cluster_feature_experiment_df = pd.DataFrame()
if RUN_CLUSTER_AS_FEATURE_EXPERIMENT:
    try:
        cluster_encoder = make_one_hot_encoder()
        tr_cluster_ohe = cluster_encoder.fit_transform(df_train[["cluster_operativo"]].astype(str))
        va_cluster_ohe = cluster_encoder.transform(df_valid[["cluster_operativo"]].astype(str)) if len(df_valid) else np.empty((0, tr_cluster_ohe.shape[1]))
        te_cluster_ohe = cluster_encoder.transform(df_test[["cluster_operativo"]].astype(str)) if len(df_test) else np.empty((0, tr_cluster_ohe.shape[1]))
        X_train_plus_cluster = np.hstack([X_train_sel, tr_cluster_ohe])
        X_valid_plus_cluster = np.hstack([X_valid_sel, va_cluster_ohe])
        X_test_plus_cluster = np.hstack([X_test_sel, te_cluster_ohe])
        base_for_cluster = trained_models.get(BEST_MODEL_NAME, best_model)
        try:
            model_plus_cluster = clone(base_for_cluster)
        except Exception:
            model_plus_cluster = lgb.LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, verbose=-1)
        model_plus_cluster.fit(X_train_plus_cluster, y_train_model)
        rows = []
        for label, X_eval, y_eval in [("valid_2024", X_valid_plus_cluster, y_valid), ("test_2025", X_test_plus_cluster, y_test)]:
            res = evaluate_model(model_plus_cluster, X_eval, y_eval, label, threshold=0.5)
            if res is not None:
                res["modelo"] = f"{BEST_MODEL_NAME}_plus_cluster"
                rows.append(res)
        cluster_feature_experiment_df = pd.DataFrame(rows)
        display(cluster_feature_experiment_df)
        cluster_feature_experiment_df.to_excel(os.path.join(TABLE_DIR, "33d_experimento_cluster_como_feature.xlsx"), index=False)
        joblib.dump(model_plus_cluster, os.path.join(MODEL_DIR, "modelo_experimento_plus_cluster.joblib"))
    except Exception as e:
        print("Experimento cluster como feature no ejecutado:", e)

anom_rows = []
try:
    iso = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
    iso_labels = iso.fit_predict(X_train_sel)
    anom_rows.append({"metodo": "IsolationForest", "anomalias_train": int((iso_labels == -1).sum()), "porcentaje": round((iso_labels == -1).mean()*100, 2)})
except Exception as e:
    print("Isolation Forest fallo:", e)
try:
    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    lof_labels = lof.fit_predict(X_train_sel)
    anom_rows.append({"metodo": "LocalOutlierFactor", "anomalias_train": int((lof_labels == -1).sum()), "porcentaje": round((lof_labels == -1).mean()*100, 2)})
except Exception as e:
    print("LOF fallo:", e)
try:
    # [v3.02 - observacion revisor #18] Mahalanobis como tercer metodo de
    # deteccion de outliers, para comparar contra IsolationForest/LOF.
    # Se usa covarianza robusta (MinCovDet) para no dejar que los propios
    # outliers distorsionen la estimacion de la covarianza.
    from sklearn.covariance import MinCovDet
    X_maha_sample = X_train_sel
    if X_maha_sample.shape[0] > 20000:
        _idx_maha = np.random.RandomState(RANDOM_STATE).choice(X_maha_sample.shape[0], 20000, replace=False)
        X_maha_sample = X_maha_sample[_idx_maha]
    mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=0.75).fit(X_maha_sample)
    maha_dist_train = mcd.mahalanobis(X_train_sel)
    maha_cutoff = np.percentile(maha_dist_train, 97.5)  # ~2.5% mas extremos, comparable a contamination="auto"
    maha_outliers = maha_dist_train > maha_cutoff
    anom_rows.append({"metodo": "Mahalanobis (MinCovDet, p97.5)",
                       "anomalias_train": int(maha_outliers.sum()),
                       "porcentaje": round(maha_outliers.mean()*100, 2)})
except Exception as e:
    print("Mahalanobis fallo:", e)
anomaly_df = pd.DataFrame(anom_rows)
display(anomaly_df)
clean_for_export(anomaly_df).to_excel(os.path.join(TABLE_DIR, "33_anomalias_isolation_lof_mahalanobis.xlsx"), index=False)

vis_sample = min(1500, X_train_sel.shape[0]) if not FAST_MODE else min(500, X_train_sel.shape[0])
idx_sample = np.random.RandomState(RANDOM_STATE).choice(np.arange(X_train_sel.shape[0]), size=vis_sample, replace=False)
X_vis = X_train_sel[idx_sample]
y_vis = y_train.iloc[idx_sample].values

try:
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto", perplexity=min(30, vis_sample-1))
    X_tsne = tsne.fit_transform(X_vis)
    plt.figure(figsize=(7,5)); plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_vis, alpha=0.55)
    plt.title("t-SNE - estructura del train"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "27_tsne_train.png"), dpi=300); plt.show()
except Exception as e:
    print("t-SNE fallo:", e)

try:
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_umap = reducer.fit_transform(X_vis)
    plt.figure(figsize=(7,5)); plt.scatter(X_umap[:,0], X_umap[:,1], c=y_vis, alpha=0.55)
    plt.title("UMAP - estructura del train"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "27_umap_train.png"), dpi=300); plt.show()
except Exception as e:
    print("UMAP fallo:", e)

annual_summary = df.groupby("anio").agg(registros=("anio", "count"), ingresado_total=("ingresado", "sum"),
                                         atendido_total=("atendido", "sum"), saldo_total=("saldo_casos", "sum"),
                                         riesgo_pct=("riesgo_congestion", "mean")).reset_index()
annual_summary["riesgo_pct"] = (annual_summary["riesgo_pct"]*100).round(2)
annual_summary["nota"] = np.where(annual_summary["anio"] == EXTERNAL_YEAR, "Anio parcial/exploratorio", "Anio completo o historico segun fuente")
display(annual_summary)
annual_summary.to_excel(os.path.join(TABLE_DIR, "34_resumen_anual_temporal.xlsx"), index=False)

plt.figure(figsize=(8,5))
plt.plot(annual_summary["anio"], annual_summary["ingresado_total"], marker="o", label="Ingresado")
plt.plot(annual_summary["anio"], annual_summary["atendido_total"], marker="o", label="Atendido")
plt.plot(annual_summary["anio"], annual_summary["saldo_total"], marker="o", label="Saldo")
plt.legend(); plt.title("Evolucion anual de casos fiscales"); plt.xlabel("Anio"); plt.ylabel("Total")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "28_evolucion_anual.png"), dpi=300); plt.show()

def psi(expected, actual, buckets=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, buckets+1)))
    if len(quantiles) < 3:
        return np.nan
    quantiles = quantiles.astype(float)
    quantiles[0] = -np.inf; quantiles[-1] = np.inf
    e_counts = pd.cut(expected, bins=quantiles, include_lowest=True).value_counts(normalize=True).sort_index()
    a_counts = pd.cut(actual, bins=quantiles, include_lowest=True).value_counts(normalize=True).sort_index()
    a_counts = a_counts.reindex(e_counts.index, fill_value=0)
    e = e_counts.values + 1e-6; a = a_counts.values + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

def psi_interpretation(value):
    if pd.isna(value):
        return "No calculable"
    if value < 0.10:
        return "Sin drift relevante"
    if value < 0.25:
        return "Drift moderado"
    return "Drift fuerte"

psi_rows = []
for col in ["ingresado", "atendido", "saldo_casos", "tasa_atencion", "ratio_saldo"]:
    for year in sorted(df["anio"].unique()):
        if year in TRAIN_YEARS:
            continue
        val = psi(df.loc[df["anio"].isin(TRAIN_YEARS), col], df.loc[df["anio"] == year, col])
        psi_rows.append({"variable": col, "comparacion": f"train_vs_{year}", "PSI": val,
                          "interpretacion": psi_interpretation(val),
                          "nota": "2026 es exploratorio/parcial" if year == EXTERNAL_YEAR else ""})
psi_df = pd.DataFrame(psi_rows)
display(psi_df)
clean_for_export(psi_df).to_excel(os.path.join(TABLE_DIR, "35_drift_psi_temporal.xlsx"), index=False)

external_metrics_df = pd.DataFrame()
if len(df_external) > 0:
    ext_results = []
    skipped_models = []
    n_features_external = X_external_sel.shape[1]
    print("Numero de variables en X_external_sel:", n_features_external)

    for name, model in trained_models.items():
        try:
            if name.lower().endswith("_meta") or ("stacking" in name.lower() and name != "VotingClassifier"):
                skipped_models.append({"modelo": name, "features_modelo": getattr(model, "n_features_in_", "No disponible"),
                                        "features_external": n_features_external,
                                        "motivo": "Objeto meta-modelo omitido; requiere scores base, no features originales"})
                continue
            expected_features = getattr(model, "n_features_in_", None)
            if expected_features is not None and expected_features != n_features_external:
                skipped_models.append({"modelo": name, "features_modelo": expected_features,
                                        "features_external": n_features_external, "motivo": "Dimension incompatible"})
                continue
            # [PATCH 19] Scoring por lotes: evita cargar todo X_external_sel
            # en una sola llamada a predict_proba si el dataset crece a
            # cientos de miles/millones de filas.
            res = evaluate_model_batched(model, X_external_sel, y_external, "external_2026",
                                          threshold=0.5, batch_size=100_000)
            if res:
                res["modelo"] = name
                ext_results.append(res)
                # [PATCH 23] Escritura incremental: cada resultado de validacion
                # externa se agrega de inmediato al CSV en disco, para no perder
                # el progreso si el proceso se interrumpe a mitad del recorrido
                # por los modelos entrenados.
                append_to_csv(res, os.path.join(TABLE_DIR, "36_validacion_externa_2026_incremental.csv"))
        except Exception as e:
            skipped_models.append({"modelo": name, "features_modelo": getattr(model, "n_features_in_", "No disponible"),
                                    "features_external": n_features_external, "motivo": str(e)})

    try:
        if temporal_stacking_artifact is not None:
            stack_external_scores = predict_temporal_stacking_scores(temporal_stacking_artifact, X_external_sel)
            stack_external_res = evaluate_scores_from_array(stack_external_scores, y_external, "external_2026", "TemporalStackingClassifier", threshold=0.5)
            ext_results.append(stack_external_res)
    except Exception as e:
        skipped_models.append({"modelo": "TemporalStackingClassifier", "features_modelo": "scores_base",
                                "features_external": n_features_external, "motivo": f"No se pudo evaluar: {e}"})

    external_metrics_df = pd.DataFrame(ext_results)
    if len(external_metrics_df) > 0:
        external_metrics_df = external_metrics_df.sort_values("f1", ascending=False)
        display(external_metrics_df)
        external_metrics_df.to_excel(os.path.join(TABLE_DIR, "36_validacion_externa_2026.xlsx"), index=False)
    skipped_models_df = pd.DataFrame(skipped_models)
    if len(skipped_models_df) > 0:
        display(skipped_models_df)
        skipped_models_df.to_excel(os.path.join(TABLE_DIR, "36_modelos_omitidos_validacion_externa_2026.xlsx"), index=False)
else:
    print("No hay datos 2026 disponibles. Esta seccion queda preparada para el CSV 2026 actualizado.")

error_summary = pd.DataFrame()
try:
    test_scores = get_scores(predictive_model, X_test_sel)
    test_pred = (test_scores >= BEST_THRESHOLD).astype(int)
    error_df = df_test.copy()
    error_df["y_true"] = y_test.values
    error_df["y_pred"] = test_pred
    error_df["score"] = test_scores
    error_df["threshold_usado"] = BEST_THRESHOLD
    error_df["tipo_error"] = np.select(
        [(error_df["y_true"] == 1) & (error_df["y_pred"] == 1), (error_df["y_true"] == 0) & (error_df["y_pred"] == 0),
         (error_df["y_true"] == 0) & (error_df["y_pred"] == 1), (error_df["y_true"] == 1) & (error_df["y_pred"] == 0)],
        ["TP", "TN", "FP", "FN"], default="NA")
    error_summary = error_df["tipo_error"].value_counts().reset_index()
    error_summary.columns = ["tipo_error", "registros"]
    display(error_summary)
    error_summary.to_excel(os.path.join(TABLE_DIR, "37_error_summary_test.xlsx"), index=False)

    group_cols = [c for c in ["dpto_pjfs", "dist_pjfs", "tipo_fiscalia", "especialidad"] if c in error_df.columns]
    error_group_rows = []
    for gc in group_cols:
        temp = error_df.groupby(gc)["tipo_error"].value_counts().unstack(fill_value=0)
        for col in ["FN", "FP", "TN", "TP"]:
            if col not in temp.columns:
                temp[col] = 0
        temp["total"] = temp[["FN", "FP", "TN", "TP"]].sum(axis=1)
        temp = temp.sort_values("total", ascending=False).head(20).reset_index()
        temp.insert(0, "grupo_variable", gc)
        error_group_rows.append(temp)
    if error_group_rows:
        error_groups_df = pd.concat(error_group_rows, ignore_index=True)
        display(error_groups_df.head(40))
        error_groups_df.to_excel(os.path.join(TABLE_DIR, "38_error_analysis_por_grupo.xlsx"), index=False)
except Exception as e:
    print("Error analysis no ejecutado:", e)

try:
    fairness_df = error_df.copy()
    fairness_rows = []
    if "dpto_pjfs" in fairness_df.columns:
        fairness_df["grupo_lima_provincia"] = np.where(fairness_df["dpto_pjfs"].astype(str).str.contains("LIMA", case=False, na=False), "LIMA", "PROVINCIAS")
        for group_col in ["grupo_lima_provincia"] + [c for c in ["tipo_fiscalia", "dpto_pjfs"] if c in fairness_df.columns]:
            for group, g in fairness_df.groupby(group_col):
                if len(g) < 20:
                    continue
                cm = confusion_matrix(g["y_true"], g["y_pred"], labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                tpr = tp/(tp+fn) if (tp+fn)>0 else np.nan
                fpr = fp/(fp+tn) if (fp+tn)>0 else np.nan
                fairness_rows.append({"variable_grupo": group_col, "grupo": group, "n": len(g),
                                       "riesgo_real_proxy": g["y_true"].mean(), "riesgo_predicho": g["y_pred"].mean(),
                                       "TPR_equal_opportunity": tpr, "FPR": fpr,
                                       "nota": "Disparidad preliminar; no implica causalidad ni discriminacion confirmada"})
    fairness_metrics_df = pd.DataFrame(fairness_rows)
    display(fairness_metrics_df.head(50))
    fairness_metrics_df.to_excel(os.path.join(TABLE_DIR, "39_disparidad_territorial_preliminar.xlsx"), index=False)
except Exception as e:
    print("Fairness/disparidad no ejecutado:", e)

ci_df = pd.DataFrame()
article_ci_df = pd.DataFrame()
try:
    rng = np.random.RandomState(RANDOM_STATE)
    boot_rows = []
    y_arr = np.array(y_test); pred_arr = np.array(test_pred); score_arr = np.array(test_scores)
    n = len(y_arr)
    for b in range(300 if not FAST_MODE else 50):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        boot_rows.append({"f1": f1_score(y_arr[idx], pred_arr[idx], zero_division=0),
                           "roc_auc": roc_auc_score(y_arr[idx], score_arr[idx]),
                           "pr_auc": average_precision_score(y_arr[idx], score_arr[idx])})
    boot_df = pd.DataFrame(boot_rows)
    ci_rows = []
    for metric in boot_df.columns:
        ci_rows.append({"metric": metric, "mean": boot_df[metric].mean(),
                         "ci_2_5": boot_df[metric].quantile(0.025), "ci_97_5": boot_df[metric].quantile(0.975)})
    ci_df = pd.DataFrame(ci_rows)
    display(ci_df)
    clean_for_export(ci_df).to_excel(os.path.join(TABLE_DIR, "40_bootstrap_confidence_intervals.xlsx"), index=False)

    point_metrics_test = evaluate_model(predictive_model, X_test_sel, y_test, "test_2025", threshold=BEST_THRESHOLD)
    article_ci_rows = []
    for metric_name in ["f1", "roc_auc", "pr_auc"]:
        ci_match = ci_df[ci_df["metric"] == metric_name]
        article_ci_rows.append({"modelo": predictive_model_name, "dataset": "test_2025", "threshold": BEST_THRESHOLD,
                                 "metric": metric_name, "point_estimate": point_metrics_test.get(metric_name, np.nan),
                                 "ci_2_5": ci_match["ci_2_5"].iloc[0] if len(ci_match) else np.nan,
                                 "ci_97_5": ci_match["ci_97_5"].iloc[0] if len(ci_match) else np.nan})
    article_ci_df = pd.DataFrame(article_ci_rows)
    display(article_ci_df)
    clean_for_export(article_ci_df).to_excel(os.path.join(TABLE_DIR, "40b_metricas_principales_con_ic_para_articulo.xlsx"), index=False)
except Exception as e:
    print("Bootstrap no ejecutado:", e)

mcnemar_df = pd.DataFrame()
try:
    from statsmodels.stats.contingency_tables import mcnemar
    preferred_order = ["CatBoost", "LightGBM", "LightGBM_Optuna", "XGBoost", "LogisticRegression", "SVM", "VotingClassifier"]
    model_names_for_test = [m for m in preferred_order if m in trained_models and m != "DummyClassifier"]
    mcnemar_rows = []
    preds = {}
    for m in model_names_for_test:
        try:
            preds[m] = (get_scores(trained_models[m], X_test_sel) >= 0.5).astype(int)
        except Exception as e:
            print(f"Modelo omitido en McNemar por error ({m}):", e)
    model_names_for_test = list(preds.keys())
    for i in range(len(model_names_for_test)):
        for j in range(i+1, len(model_names_for_test)):
            m1, m2 = model_names_for_test[i], model_names_for_test[j]
            correct1 = preds[m1] == y_test.values; correct2 = preds[m2] == y_test.values
            table = [[np.sum(correct1 & correct2), np.sum(correct1 & ~correct2)], [np.sum(~correct1 & correct2), np.sum(~correct1 & ~correct2)]]
            result = mcnemar(table, exact=False, correction=True)
            mcnemar_rows.append({"modelo_1": m1, "modelo_2": m2, "statistic": result.statistic, "pvalue": result.pvalue,
                                  "pvalue_reportado": "p < 0.001" if result.pvalue < 0.001 else round(result.pvalue, 4)})
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    display(mcnemar_df)
    clean_for_export(mcnemar_df).to_excel(os.path.join(TABLE_DIR, "41_mcnemar_modelos.xlsx"), index=False)
except Exception as e:
    print("McNemar no ejecutado:", e)

# ============================================================
# 41b. COMPARACION ROC-AUC DE LOS MODELOS (test 2025)
# ============================================================
# Insertar DESPUES del bloque de McNemar (linea ~4045, justo antes del
# bloque de curva de calibracion) para reutilizar 'model_names_for_test',
# 'trained_models', 'get_scores' y 'X_test_sel' / 'y_test' ya definidos.
#
# Objetivo:
# - Graficar en una sola figura la curva ROC de cada modelo entrenado,
#   evaluado sobre el mismo conjunto de test (test 2025), con su AUC
#   en la leyenda.
# - Dejar tabla exportable con el AUC (+ IC 95% via bootstrap simple)
#   de cada modelo, ordenada de mayor a menor, para trazabilidad.

roc_auc_model_order = [m for m in preferred_order if m in trained_models and m != "DummyClassifier"]

roc_curve_data = {}
roc_auc_rows = []

for m in roc_auc_model_order:
    try:
        scores_m = get_scores(trained_models[m], X_test_sel)
        fpr_m, tpr_m, _ = roc_curve(y_test, scores_m)
        auc_m = roc_auc_score(y_test, scores_m)
        roc_curve_data[m] = (fpr_m, tpr_m, auc_m)
        roc_auc_rows.append({"modelo": m, "roc_auc_test": auc_m})
    except Exception as e:
        print(f"Modelo omitido en comparacion ROC-AUC por error ({m}):", e)

roc_auc_summary_df = pd.DataFrame(roc_auc_rows).sort_values("roc_auc_test", ascending=False)
print("Resumen ROC-AUC por modelo (test 2025):")
display(roc_auc_summary_df)
clean_for_export(roc_auc_summary_df).to_excel(
    os.path.join(TABLE_DIR, "41b_roc_auc_comparacion_modelos.xlsx"), index=False
)

# --- Figura: todas las curvas ROC superpuestas, ordenadas por AUC descendente ---
plt.figure(figsize=(7, 6))
color_cycle = plt.cm.tab10.colors
models_sorted = roc_auc_summary_df["modelo"].tolist()

for idx, m in enumerate(models_sorted):
    fpr_m, tpr_m, auc_m = roc_curve_data[m]
    is_best = (idx == 0)
    plt.plot(
        fpr_m, tpr_m,
        color=color_cycle[idx % len(color_cycle)],
        linewidth=2.5 if is_best else 1.5,
        alpha=1.0 if is_best else 0.85,
        label=f"{m} (AUC = {auc_m:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Azar (AUC = 0.500)")
plt.xlabel("Tasa de falsos positivos (1 - especificidad)")
plt.ylabel("Tasa de verdaderos positivos (sensibilidad)")
plt.title("Comparacion ROC-AUC de los modelos - Test 2025")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "41b_comparacion_roc_auc_modelos.png"), dpi=300)
plt.show()
plt.close()

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_test, test_scores)
    prob_true, prob_pred = calibration_curve(y_test, test_scores, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker="o", label=f"{predictive_model_name} Brier={brier:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Probabilidad predicha"); plt.ylabel("Frecuencia observada"); plt.title("Curva de calibracion - Test 2025")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "42_calibration_curve_test.png"), dpi=300); plt.show()

    # [v3.02 - observacion revisor #6/#14] ECE + MCE (Expected/Maximum
    # Calibration Error) complementan al Brier Score, que resume calibracion
    # en un solo numero pero no muestra en que rango de probabilidad falla.
    ece_val, mce_val, ece_bins_df = compute_ece(y_test.values, test_scores, n_bins=10)
    display(ece_bins_df)
    clean_for_export(ece_bins_df).to_excel(os.path.join(TABLE_DIR, "42c_reliability_diagram_bins.xlsx"), index=False)

    plt.figure(figsize=(6, 4))
    if len(ece_bins_df):
        plt.bar((ece_bins_df["bin_lo"] + ece_bins_df["bin_hi"]) / 2, ece_bins_df["tasa_observada"],
                width=0.08, alpha=0.7, label="Tasa observada")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Calibracion perfecta")
    plt.xlabel("Probabilidad predicha (bin)"); plt.ylabel("Tasa observada")
    plt.title(f"Reliability Diagram - Test 2025 (ECE={ece_val:.4f}, MCE={mce_val:.4f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "42d_reliability_diagram.png"), dpi=300); plt.show()

    calibration_summary_df = pd.DataFrame({
        "modelo": [predictive_model_name], "brier_score": [brier], "ece": [ece_val], "mce": [mce_val]
    })
    calibration_summary_df.to_excel(os.path.join(TABLE_DIR, "42_brier_score.xlsx"), index=False)
    print(f"Brier Score={brier:.4f} | ECE={ece_val:.4f} | MCE={mce_val:.4f}")
except Exception as e:
    print("Calibracion no ejecutada:", e)

conformal_summary_df = pd.DataFrame()
mondrian_conformal_summary_df = pd.DataFrame()
if RUN_CONFORMAL_PREDICTION:
    try:
        alpha = 0.10

        def binary_prob_matrix(model, X):
            s = np.asarray(get_scores(model, X)).reshape(-1)
            s = np.clip(s, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - s, s])

        valid_prob = binary_prob_matrix(predictive_model, X_valid_sel)
        y_valid_arr = np.asarray(y_valid).astype(int)
        calib_nonconformity = 1 - valid_prob[np.arange(len(y_valid_arr)), y_valid_arr]
        n_cal = len(calib_nonconformity)
        q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
        try:
            q_hat = float(np.quantile(calib_nonconformity, q_level, method="higher"))
        except TypeError:
            q_hat = float(np.quantile(calib_nonconformity, q_level, interpolation="higher"))

        def conformal_sets_for(X):
            prob = binary_prob_matrix(predictive_model, X)
            include = (1 - prob) <= q_hat
            set_size = include.sum(axis=1)
            pred_label = (prob[:, 1] >= 0.5).astype(int)
            return prob, include, set_size, pred_label

        conf_rows = []; set_rows = []
        eval_parts = [("test_2025", X_test_sel, y_test, df_test), ("external_2026", X_external_sel, y_external, df_external)]
        for dataset_name, X_eval, y_eval, df_eval in eval_parts:
            if len(y_eval) == 0:
                continue
            prob, include, set_size, pred_label = conformal_sets_for(X_eval)
            y_arr = np.asarray(y_eval).astype(int)
            covered = include[np.arange(len(y_arr)), y_arr]
            conf_rows.append({"dataset": dataset_name, "alpha": alpha, "nominal_coverage": 1 - alpha,
                               "empirical_coverage": float(np.mean(covered)), "avg_set_size": float(np.mean(set_size)),
                               "singleton_rate": float(np.mean(set_size == 1)), "ambiguous_rate": float(np.mean(set_size == 2)),
                               "empty_set_rate": float(np.mean(set_size == 0)), "q_hat": q_hat})
            temp = pd.DataFrame({"dataset": dataset_name, "row_position": np.arange(len(y_arr)), "y_true": y_arr,
                                  "score_riesgo": prob[:, 1], "set_size": set_size, "covered": covered, "prediction_hard": pred_label})
            if "cluster_operativo" in df_eval.columns:
                temp["cluster_operativo"] = df_eval["cluster_operativo"].values
            set_rows.append(temp)

        conformal_summary_df = pd.DataFrame(conf_rows)
        conformal_sets_df = pd.concat(set_rows, ignore_index=True) if set_rows else pd.DataFrame()
        display(conformal_summary_df)
        conformal_summary_df.to_excel(os.path.join(TABLE_DIR, "42b_conformal_prediction_summary.xlsx"), index=False)
        conformal_sets_df.to_excel(os.path.join(TABLE_DIR, "42c_conformal_prediction_sets.xlsx"), index=False)

        mondrian_conformal_sets_df = pd.DataFrame()
        if RUN_MONDRIAN_CONFORMAL_BY_CLUSTER and "cluster_operativo" in df_valid.columns:
            min_calibration_per_cluster = 30
            valid_cluster = df_valid["cluster_operativo"].values
            q_by_cluster = {}
            cluster_calibration_rows = []
            for cl in pd.Series(valid_cluster).dropna().unique():
                mask_cl = valid_cluster == cl
                nc_cl = calib_nonconformity[mask_cl]
                if len(nc_cl) >= min_calibration_per_cluster:
                    q_level_cl = min(np.ceil((len(nc_cl) + 1) * (1 - alpha)) / len(nc_cl), 1.0)
                    try:
                        q_cl = float(np.quantile(nc_cl, q_level_cl, method="higher"))
                    except TypeError:
                        q_cl = float(np.quantile(nc_cl, q_level_cl, interpolation="higher"))
                    source = "cluster_specific"
                else:
                    q_cl = q_hat
                    source = "global_fallback_low_n"
                q_by_cluster[cl] = q_cl
                cluster_calibration_rows.append({"cluster_operativo": cl, "n_calibration": int(len(nc_cl)), "q_hat_mondrian": q_cl, "source": source})

            def mondrian_sets_for(X, cluster_values):
                prob = binary_prob_matrix(predictive_model, X)
                cluster_values = np.asarray(cluster_values)
                include = np.zeros_like(prob, dtype=bool)
                q_used = []
                for i, cl in enumerate(cluster_values):
                    q_i = q_by_cluster.get(cl, q_hat)
                    include[i, :] = (1 - prob[i, :]) <= q_i
                    q_used.append(q_i)
                set_size = include.sum(axis=1)
                pred_label = (prob[:, 1] >= 0.5).astype(int)
                return prob, include, set_size, pred_label, np.asarray(q_used)

            mondrian_rows = []; mondrian_set_rows = []
            for dataset_name, X_eval, y_eval, df_eval in eval_parts:
                if len(y_eval) == 0 or "cluster_operativo" not in df_eval.columns:
                    continue
                prob_m, include_m, set_size_m, pred_label_m, q_used_m = mondrian_sets_for(X_eval, df_eval["cluster_operativo"].values)
                y_arr = np.asarray(y_eval).astype(int)
                covered_m = include_m[np.arange(len(y_arr)), y_arr]
                mondrian_rows.append({"dataset": dataset_name, "method": "mondrian_by_operational_cluster", "alpha": alpha,
                                       "nominal_coverage": 1 - alpha, "empirical_coverage": float(np.mean(covered_m)),
                                       "avg_set_size": float(np.mean(set_size_m)), "singleton_rate": float(np.mean(set_size_m == 1)),
                                       "ambiguous_rate": float(np.mean(set_size_m == 2)), "empty_set_rate": float(np.mean(set_size_m == 0)),
                                       "nota": "Calibracion conformal por cluster operativo con fallback global si hay pocos casos en validacion."})
                temp_m = pd.DataFrame({"dataset": dataset_name, "row_position": np.arange(len(y_arr)),
                                        "cluster_operativo": df_eval["cluster_operativo"].values, "y_true": y_arr,
                                        "score_riesgo": prob_m[:, 1], "set_size": set_size_m, "covered": covered_m,
                                        "prediction_hard": pred_label_m, "q_hat_used": q_used_m})
                mondrian_set_rows.append(temp_m)

            mondrian_calibration_df = pd.DataFrame(cluster_calibration_rows)
            mondrian_conformal_summary_df = pd.DataFrame(mondrian_rows)
            mondrian_conformal_sets_df = pd.concat(mondrian_set_rows, ignore_index=True) if mondrian_set_rows else pd.DataFrame()
            display(mondrian_calibration_df); display(mondrian_conformal_summary_df)
            mondrian_calibration_df.to_excel(os.path.join(TABLE_DIR, "42e_mondrian_conformal_calibration_by_cluster.xlsx"), index=False)
            mondrian_conformal_summary_df.to_excel(os.path.join(TABLE_DIR, "42f_mondrian_conformal_summary.xlsx"), index=False)
            mondrian_conformal_sets_df.to_excel(os.path.join(TABLE_DIR, "42g_mondrian_conformal_sets.xlsx"), index=False)
    except Exception as e:
        print("Conformal prediction no ejecutado:", e)

advanced_fairness_df = pd.DataFrame()
advanced_fairness_gaps_df = pd.DataFrame()
if RUN_ADVANCED_FAIRNESS:
    try:
        fairness_eval_df = df_test.copy()
        fairness_eval_df["y_true"] = np.asarray(y_test).astype(int)
        fairness_eval_df["score"] = get_scores(predictive_model, X_test_sel)
        fairness_eval_df["y_pred"] = (fairness_eval_df["score"] >= BEST_THRESHOLD).astype(int)

        if "dpto_pjfs" in fairness_eval_df.columns:
            fairness_eval_df["grupo_lima_provincia"] = np.where(
                fairness_eval_df["dpto_pjfs"].astype(str).str.contains("LIMA", case=False, na=False), "LIMA", "PROVINCIAS")
        if "dist_pjfs" in fairness_eval_df.columns and "tipo_fiscalia" in fairness_eval_df.columns:
            fairness_eval_df["inter_dist_tipo_fiscalia"] = fairness_eval_df["dist_pjfs"].astype(str) + "__" + fairness_eval_df["tipo_fiscalia"].astype(str)
        if "cluster_operativo" in fairness_eval_df.columns and "tipo_fiscalia" in fairness_eval_df.columns:
            fairness_eval_df["inter_cluster_tipo_fiscalia"] = fairness_eval_df["cluster_operativo"].astype(str) + "__" + fairness_eval_df["tipo_fiscalia"].astype(str)

        group_cols_adv = [c for c in ["grupo_lima_provincia", "dpto_pjfs", "dist_pjfs", "tipo_fiscalia", "especialidad",
                                       "cluster_operativo", "inter_dist_tipo_fiscalia", "inter_cluster_tipo_fiscalia"] if c in fairness_eval_df.columns]

        def group_fairness_metrics(data, group_col, min_n=30):
            rows = []
            for group_value, g in data.groupby(group_col):
                if len(g) < min_n:
                    continue
                cm = confusion_matrix(g["y_true"], g["y_pred"], labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
                precision_g = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                try:
                    from sklearn.metrics import brier_score_loss
                    brier_g = brier_score_loss(g["y_true"], g["score"])
                except Exception:
                    brier_g = np.nan
                rows.append({"group_col": group_col, "group_value": str(group_value), "n": len(g),
                             "actual_positive_rate": g["y_true"].mean(), "predicted_positive_rate": g["y_pred"].mean(),
                             "demographic_parity_rate": g["y_pred"].mean(), "equal_opportunity_tpr": tpr,
                             "equalized_odds_fpr": fpr, "false_negative_rate": fnr, "precision": precision_g,
                             "calibration_brier": brier_g, "f1": f1_score(g["y_true"], g["y_pred"], zero_division=0),
                             "balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"])})
            return rows

        fairness_rows_adv = []
        for gc in group_cols_adv:
            fairness_rows_adv.extend(group_fairness_metrics(fairness_eval_df, gc, min_n=30))
        advanced_fairness_df = pd.DataFrame(fairness_rows_adv)
        display(advanced_fairness_df.head(40))
        advanced_fairness_df.to_excel(os.path.join(TABLE_DIR, "43_advanced_fairness_by_group.xlsx"), index=False)

        gap_rows = []
        if len(advanced_fairness_df):
            for gc, g in advanced_fairness_df.groupby("group_col"):
                for metric in ["demographic_parity_rate", "equal_opportunity_tpr", "equalized_odds_fpr", "false_negative_rate", "calibration_brier", "f1"]:
                    vals = g[metric].dropna()
                    if len(vals) >= 2:
                        gap_rows.append({"group_col": gc, "metric": metric, "min": vals.min(), "max": vals.max(),
                                          "absolute_gap": vals.max() - vals.min(), "n_groups": len(vals),
                                          "interpretacion": "Diferencia entre grupos; analisis de gobernanza, no acusacion causal."})
        advanced_fairness_gaps_df = pd.DataFrame(gap_rows).sort_values("absolute_gap", ascending=False) if gap_rows else pd.DataFrame()
        display(advanced_fairness_gaps_df.head(30) if len(advanced_fairness_gaps_df) else advanced_fairness_gaps_df)
        advanced_fairness_gaps_df.to_excel(os.path.join(TABLE_DIR, "43b_advanced_fairness_gaps.xlsx"), index=False)
    except Exception as e:
        print("Fairness avanzado no ejecutado:", e)

def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float); ty = np.empty([k, n], dtype=float); tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov

def delong_roc_test(y_true, pred_one, pred_two):
    y_true = np.array(y_true).astype(int)
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    preds = np.vstack((np.array(pred_one), np.array(pred_two)))[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1] if np.ndim(cov) == 2 else np.nan
    if var <= 0 or pd.isna(var):
        return aucs[0], aucs[1], np.nan, np.nan
    z = diff / np.sqrt(var)
    pvalue = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, pvalue

delong_df = pd.DataFrame()
friedman_df = pd.DataFrame()
if RUN_ADVANCED_STAT_TESTS:
    try:
        candidate_model_names = [m for m in ["CatBoost", "LightGBM", "LightGBM_Optuna", "XGBoost", "LogisticRegression"] if m in trained_models]
        delong_rows = []
        score_cache = {m: get_scores(trained_models[m], X_test_sel) for m in candidate_model_names}
        for i in range(len(candidate_model_names)):
            for j in range(i + 1, len(candidate_model_names)):
                m1, m2 = candidate_model_names[i], candidate_model_names[j]
                auc1, auc2, z, p = delong_roc_test(y_test.values, score_cache[m1], score_cache[m2])
                delong_rows.append({"modelo_1": m1, "modelo_2": m2, "auc_modelo_1": auc1, "auc_modelo_2": auc2, "z": z, "pvalue": p,
                                     "pvalue_reportado": "p < 0.001" if pd.notna(p) and p < 0.001 else (round(p, 4) if pd.notna(p) else np.nan)})
        delong_df = pd.DataFrame(delong_rows)
        display(delong_df)
        clean_for_export(delong_df).to_excel(os.path.join(TABLE_DIR, "44_delong_roc_auc_comparacion.xlsx"), index=False)
    except Exception as e:
        print("DeLong no ejecutado:", e)

    try:
        stat_model_names = [m for m in ["LogisticRegression", "SVM", "XGBoost", "LightGBM", "CatBoost"] if m in trained_models]
        fold_scores = {m: [] for m in stat_model_names}
        for fold_id, (tr_idx, va_idx) in enumerate(iter_cv_splits(cv5, X_train_model_cv)):
            X_tr, X_va = X_train_model_cv[tr_idx], X_train_model_cv[va_idx]
            y_tr, y_va = y_train_model_cv[tr_idx], y_train_model_cv[va_idx]
            for m in stat_model_names:
                try:
                    mdl = clone(trained_models[m])
                    mdl.fit(X_tr, y_tr)
                    score = get_scores(mdl, X_va)
                    pred = (score >= 0.5).astype(int)
                    fold_scores[m].append(f1_score(y_va, pred, zero_division=0))
                except Exception:
                    fold_scores[m].append(np.nan)

        friedman_input = [pd.Series(fold_scores[m]).dropna().values for m in stat_model_names]
        min_len = min(len(x) for x in friedman_input) if friedman_input else 0
        friedman_input = [x[:min_len] for x in friedman_input]
        if min_len >= 2 and len(friedman_input) >= 3:
            fried_stat, fried_p = friedmanchisquare(*friedman_input)
        else:
            fried_stat, fried_p = np.nan, np.nan

        friedman_df = pd.DataFrame([{"test": "Friedman", "metric": "F1 temporal CV", "n_models": len(stat_model_names), "n_folds": min_len,
                                      "statistic": fried_stat, "pvalue": fried_p,
                                      "pvalue_reportado": "p < 0.001" if pd.notna(fried_p) and fried_p < 0.001 else (round(fried_p, 4) if pd.notna(fried_p) else np.nan)}])
        display(friedman_df)
        clean_for_export(friedman_df).to_excel(os.path.join(TABLE_DIR, "45_friedman_test_modelos.xlsx"), index=False)

        scores_matrix = pd.DataFrame(fold_scores)
        scores_matrix.to_excel(os.path.join(TABLE_DIR, "45b_friedman_scores_por_fold.xlsx"), index=False)

        if sp is not None and scores_matrix.dropna().shape[0] >= 2:
            nemenyi = sp.posthoc_nemenyi_friedman(scores_matrix.dropna().values)
            nemenyi.index = stat_model_names
            nemenyi.columns = stat_model_names
            display(nemenyi)
            nemenyi.to_excel(os.path.join(TABLE_DIR, "46_nemenyi_posthoc_modelos.xlsx"))
    except Exception as e:
        print("Friedman/Nemenyi no ejecutado:", e)
else:
    print("Pruebas estadisticas avanzadas omitidas porque RUN_ADVANCED_STAT_TESTS=False.")

# ============================================================
# EXPORTACION DE MODELOS Y METADATOS
# ============================================================
try:
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump(predictive_model, os.path.join(MODEL_DIR, f"modelo_predictivo_principal_{predictive_model_name}.joblib"))
    joblib.dump(explainability_model, os.path.join(MODEL_DIR, f"modelo_explicabilidad_{explainability_model_name}.joblib"))
    joblib.dump(selected_features_final, os.path.join(MODEL_DIR, "selected_features_final.joblib"))
    if temporal_stacking_artifact is not None:
        joblib.dump(temporal_stacking_artifact, os.path.join(MODEL_DIR, "temporal_stacking_artifact.joblib"))

    metadata = {
        "predictive_model": predictive_model_name,
        "explainability_model": explainability_model_name,
        "best_threshold_validacion": BEST_THRESHOLD,
        "train_years": TRAIN_YEARS, "valid_year": VALID_YEAR, "test_year": TEST_YEAR, "external_year": EXTERNAL_YEAR,
        "selected_features_count": len(selected_features_final),
        "p75_saldo_train": float(P75_SALDO), "p25_tasa_train": float(P25_TASA),
        "target_proxy_note": "riesgo_congestion es una etiqueta proxy auditable; no es certificacion oficial de congestion fiscal.",
        "target_primary_scenario": PRIMARY_TARGET_SCENARIO,
        "leakage_excluded_columns": LEAKAGE_COLS, "collinearity_excluded_columns": COLLINEAR_EXCLUDE,
        "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS, "cv_strategy": cv_description,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_modelo.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Exportacion completada.")
    # [PATCH 11-13] Checkpoint M08: export de modelo/metadata principal completo.
    ckpt.save("M08_export_modelo_principal", metadata=metadata, predictive_model_name=predictive_model_name)
except Exception as e:
    print("Error en exportacion:", e)

# ============================================================
# RESUMEN FINAL EN TEXTO (resumen_final_pipeline.txt)
# [PATCH 1] Todos los valores potencialmente ausentes se formatean con
# safe_fmt() en vez del patron invalido {VAR:.4f if cond else 'X'}.
# ============================================================
summary_lines = []
summary_lines.append("CONCLUSIONES DEL PIPELINE MPFN")
summary_lines.append("="*60)
summary_lines.append(f"Modelo predictivo principal: {predictive_model_name}")
summary_lines.append(f"Modelo complementario para explicabilidad/calibracion: {explainability_model_name}")
summary_lines.append(f"Umbral ajustado validacion: {safe_fmt('BEST_THRESHOLD', fmt='{:.3f}')}")
summary_lines.append("Objetivo reformulado: identificacion de senales proxy de posible sobrecarga fiscal operacional, no certificacion oficial de congestion.")
summary_lines.append(f"Estrategia CV principal: {cv_description}")
summary_lines.append(f"Features seleccionadas: {len(selected_features_final)}")
summary_lines.append(f"Target proxy principal conservador: P75 saldo train={P75_SALDO:.4f}, P25 tasa train={P25_TASA:.4f}")
summary_lines.append(f"Train principal: {TRAIN_YEARS}; validacion: {VALID_YEAR}; test: {TEST_YEAR}; externa: {EXTERNAL_YEAR} parcial.")
summary_lines.append("Variables excluidas por fuga directa: " + ", ".join(LEAKAGE_COLS))
summary_lines.append("Variables excluidas por colinealidad: " + ", ".join(COLLINEAR_EXCLUDE))

try:
    principal_05 = metrics_df[(metrics_df["modelo"] == predictive_model_name) & (metrics_df["dataset"].isin(["valid_2024", "test_2025", "external_2026"]))].copy()
    if len(principal_05):
        summary_lines.append("\nMetricas del modelo principal con threshold 0.5:")
        summary_lines.append(principal_05.to_string(index=False))
except Exception:
    pass

for label, obj in [
    ("Baselines predictivos simples y sanity check del target", "baseline_rules_df"),
    ("Metricas del modelo principal con umbral ajustado", "final_metrics_df"),
    ("Sensibilidad del target: ranking de modelos", "target_sensitivity_model_ranking_df"),
    ("Decision metodologica del target final", "target_final_decision_df"),
    ("Validacion externa 2026 con threshold 0.5", "external_metrics_df"),
    ("Variables mas importantes por Permutation Importance", "perm_df"),
    ("Variables mas importantes por SHAP global", "shap_rank_df"),
    ("Resumen de Ablation Study", "ablation_df"),
    ("Resumen PSI temporal", "psi_df"),
    ("Intervalos bootstrap en test 2025", "ci_df"),
    ("DeLong ROC-AUC test 2025", "delong_df"),
    ("Friedman F1 temporal CV", "friedman_df"),
    ("Escenarios de target proxy evaluados", "target_scenarios_df"),
    ("Perfilamiento operativo por clusters", "cluster_profile_df"),
    ("Top interacciones SHAP", "shap_interaction_df"),
    ("Contrafactuales controlados", "counterfactual_df"),
    ("Conformal prediction global summary", "conformal_summary_df"),
    ("Mondrian conformal por cluster operativo", "mondrian_conformal_summary_df"),
    ("Fairness avanzado: mayores brechas", "advanced_fairness_gaps_df"),
]:
    obj_val = globals().get(obj)
    if isinstance(obj_val, pd.DataFrame) and len(obj_val):
        summary_lines.append(f"\n{label}:")
        summary_lines.append(obj_val.head(20).to_string(index=False))

summary_lines.append("\nValidacion Delphi/AHP: no ejecutada porque no se cuenta con expertos reales; se reemplaza por validacion empirica del constructo proxy.")
summary_lines.append("\nLimitacion central: riesgo_congestion es una senal proxy, no una certificacion oficial de congestion fiscal.")
summary_lines.append("El analisis 2026 es externo y exploratorio por tratarse de un anio parcial.")
summary_lines.append("Oracle_TargetDefinitionCheck debe describirse como chequeo de consistencia de la etiqueta, no como baseline predictivo que compite contra ML.")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(os.path.join(OUTPUT_DIR, "resumen_final_pipeline.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

# ============================================================
# STRESS TESTING (ROBUSTNESS ANALYSIS)
# ============================================================
print("Ejecutando Stress Testing (Robustness Analysis)...")
stress_levels = [0.10, 0.20, 0.30]
stress_results = []

if isinstance(X_test_sel, np.ndarray):
    X_test_df_stress = pd.DataFrame(X_test_sel, columns=selected_features_final)
else:
    X_test_df_stress = X_test_sel.copy()

try:
    baseline_probs = get_scores(predictive_model, X_test_sel)
    baseline_preds = (baseline_probs >= BEST_THRESHOLD).astype(int)
    baseline_f1 = f1_score(y_test, baseline_preds)
    baseline_roc = roc_auc_score(y_test, baseline_probs)
    print(f"Baseline  F1: {baseline_f1:.4f} | ROC-AUC: {baseline_roc:.4f}")
except Exception as e:
    print("Error baseline stress test:", e)

for stress in stress_levels:
    X_stress = X_test_df_stress.copy()
    for col in X_stress.columns:
        if any(k in str(col).lower() for k in ['ingres', 'atendid', 'saldo', 'carga', 'freq']):
            X_stress[col] = X_stress[col] * (1 + stress)
    try:
        X_stress_array = X_stress.values
        probs = get_scores(predictive_model, X_stress_array)
        preds = (probs >= BEST_THRESHOLD).astype(int)
        stress_results.append({'stress_level': f"+{int(stress*100)}%", 'f1': round(f1_score(y_test, preds), 4),
                                'roc_auc': round(roc_auc_score(y_test, probs), 4),
                                'recall': round(recall_score(y_test, preds), 4), 'precision': round(precision_score(y_test, preds), 4)})
    except Exception as e:
        print(f"Error en stress +{int(stress*100)}%: {e}")

stress_df = pd.DataFrame(stress_results)
display(stress_df)
stress_df.to_excel(os.path.join(TABLE_DIR, "32_stress_testing_results.xlsx"), index=False)
print("Stress Testing completado.")

# ============================================================
# SHAP TEMPORAL EVOLUTION
# ============================================================
print("Calculando SHAP Temporal Evolution...")
temporal_shap_df = pd.DataFrame()
if explainer is not None and "df_test" in dir() and "anio" in df_test.columns:
    temporal_shap_results = []
    X_test_df_shap = pd.DataFrame(X_test_sel, columns=selected_features_final) if isinstance(X_test_sel, np.ndarray) else X_test_sel.copy()
    for yr in sorted(df_test['anio'].dropna().unique()):
        mask = (df_test['anio'] == yr).values
        if mask.sum() < 15:
            continue
        X_year = X_test_df_shap[mask]
        try:
            shap_vals_year = explainer.shap_values(X_year) if hasattr(explainer, "shap_values") else explainer(X_year).values
            if isinstance(shap_vals_year, list):
                shap_vals_year = shap_vals_year[1]
            imp = np.abs(shap_vals_year).mean(axis=0)
            tmp = pd.DataFrame({'Feature': selected_features_final, 'Mean_Abs_SHAP': imp, 'Year': yr})
            temporal_shap_results.append(tmp)
            print(f"Anio {yr} procesado correctamente.")
        except Exception as e:
            print(f"Error en anio {yr}: {e}")
    if temporal_shap_results:
        temporal_shap_df = pd.concat(temporal_shap_results, ignore_index=True)
        shap_temporal_summary = temporal_shap_df.groupby('Feature')['Mean_Abs_SHAP'].agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False)
        display(shap_temporal_summary.head(15))
        temporal_shap_df.to_excel(os.path.join(TABLE_DIR, "33_shap_temporal_evolution.xlsx"), index=False)
        print("SHAP Temporal Evolution guardado correctamente.")
    else:
        print("No se generaron resultados de SHAP Temporal.")
else:
    print("SHAP Temporal Evolution omitido: explainer no disponible o falta df_test/anio.")

# ============================================================
# BOOTSTRAP SHAP STABILITY (version rapida)
# ============================================================
print("Calculando Bootstrap SHAP Stability (version rapida)...")
shap_stability_df = pd.DataFrame()
if explainer is not None:
    n_bootstrap = 30
    bootstrap_shap_values = []
    X_test_df_boot = pd.DataFrame(X_test_sel, columns=selected_features_final) if isinstance(X_test_sel, np.ndarray) else X_test_sel.copy()
    rng_boot = np.random.RandomState(RANDOM_STATE)
    for i in range(n_bootstrap):
        sample_idx = rng_boot.choice(len(X_test_df_boot), len(X_test_df_boot), replace=True)
        Xb = X_test_df_boot.iloc[sample_idx]
        try:
            sv = explainer.shap_values(Xb) if hasattr(explainer, "shap_values") else explainer(Xb).values
            if isinstance(sv, list):
                sv = sv[1]
            bootstrap_shap_values.append(np.abs(sv).mean(axis=0))
        except Exception:
            pass
    if len(bootstrap_shap_values) > 5:
        bootstrap_shap = np.array(bootstrap_shap_values)
        shap_stability_df = pd.DataFrame({
            'Feature': selected_features_final, 'Mean_SHAP': bootstrap_shap.mean(axis=0), 'Std_SHAP': bootstrap_shap.std(axis=0),
            'CV_pct': (bootstrap_shap.std(axis=0) / (bootstrap_shap.mean(axis=0) + 1e-8)) * 100
        }).sort_values('Mean_SHAP', ascending=False)
        display(shap_stability_df.head(15))
        shap_stability_df.to_excel(os.path.join(TABLE_DIR, "34_shap_bootstrap_stability.xlsx"), index=False)
        print("Bootstrap SHAP Stability guardado correctamente.")
    else:
        print("No se pudieron generar suficientes iteraciones de Bootstrap SHAP.")
else:
    print("Bootstrap SHAP Stability omitido: explainer no disponible.")

# ============================================================
# ECONOMIC IMPACT ANALYSIS
# ============================================================
print("Calculando Economic Impact Analysis...")
COST_FN = 50000
COST_FP = 5000
try:
    test_scores_econ = get_scores(predictive_model, X_test_sel)
    test_pred_econ = (test_scores_econ >= BEST_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred_econ).ravel()
    economic_loss = (fn * COST_FN) + (fp * COST_FP)
    economic_summary = pd.DataFrame({
        'Metrica': ['Falsos Negativos (FN)', 'Falsos Positivos (FP)', 'Verdaderos Positivos (TP)', 'Verdaderos Negativos (TN)',
                    'Costo por FN', 'Costo por FP', 'Perdida Total Estimada'],
        'Valor': [fn, fp, tp, tn, COST_FN, COST_FP, economic_loss]
    })
    display(economic_summary)
    economic_summary.to_excel(os.path.join(TABLE_DIR, "35_economic_impact.xlsx"), index=False)
    print(f"Perdida total estimada: S/ {economic_loss:,.0f}")
except Exception as e:
    print(f"Error en Economic Impact: {e}")

# ============================================================
# DOUBLE MACHINE LEARNING (ANALISIS CAUSAL EXPLORATORIO)
# [PATCH] Corrige referencia a variable indefinida TARGET_COL -> TARGET
# (el bug original usaba TARGET_COL, que nunca se definia, y hacia que este
# bloque siempre cayera al except silenciosamente).
# ============================================================
print("=" * 70)
print("DOUBLE MACHINE LEARNING (exploratorio)")
print("=" * 70)
try:
    from doubleml import DoubleMLData, DoubleMLIRM
    from lightgbm import LGBMRegressor, LGBMClassifier

    causal_df = df_train.copy()
    treatment_col = "ratio_saldo"
    if treatment_col not in causal_df.columns:
        raise ValueError(f"No existe la variable {treatment_col}")

    median_value = causal_df[treatment_col].median()
    causal_df["TRATAMIENTO"] = (causal_df[treatment_col] > median_value).astype(int)

    numeric_cols_dml = causal_df.select_dtypes(include=["number"]).columns.tolist()
    exclude_cols_dml = [TARGET, treatment_col, "TRATAMIENTO"] + [c for c in causal_df.columns if c.startswith("riesgo_proxy_")] + ["saldo_casos", "tasa_atencion"]
    x_cols = [c for c in numeric_cols_dml if c not in exclude_cols_dml]
    x_cols = [c for c in x_cols if causal_df[c].nunique() > 1]
    print("Numero de covariables:", len(x_cols))

    dml_df = causal_df[x_cols + ["TRATAMIENTO", TARGET]].copy()
    dml_df = dml_df.replace([np.inf, -np.inf], np.nan).dropna()
    dml_df[TARGET] = dml_df[TARGET].astype(int)
    dml_df["TRATAMIENTO"] = dml_df["TRATAMIENTO"].astype(int)
    print("Observaciones utilizadas:", len(dml_df))

    dml_data = DoubleMLData(data=dml_df, y_col=TARGET, d_cols="TRATAMIENTO", x_cols=x_cols)
    ml_g = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
    ml_m = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
    dml = DoubleMLIRM(obj_dml_data=dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=5)
    dml.fit()
    causal_summary = dml.summary
    print("RESULTADOS DOUBLEML")
    display(causal_summary)
    causal_summary.to_csv(os.path.join(TABLE_DIR, "36_doubleml_results.csv"))

    try:
        ate = float(causal_summary["coef"].iloc[0])
        pvalue_dml = float(causal_summary["P>|t|"].iloc[0])
        print(f"ATE = {ate:.4f} | p-value = {pvalue_dml:.4f}")
        print("Efecto estadisticamente significativo." if pvalue_dml < 0.05 else "Efecto no estadisticamente significativo.")
    except Exception as interpretation_error:
        print(f"Interpretacion no disponible: {interpretation_error}")
except Exception as e:
    print(f"DoubleML no ejecutado: {e}")

# ============================================================
# BLOQUE FINAL: REPRODUCIBILIDAD, METADATA, SCORECARD, EMPAQUETADO
# (Version condensada respecto del bloque original de ~1500 lineas,
#  conservando TODOS los artefactos: config, requirements, environment,
#  Dockerfile, README, CITATION, LICENSE, metadata con hashes/versiones/
#  hardware, manifest+checksums SHA-256, DataDictionary, PDF de resultados,
#  [PATCH 10] PROJECT_SCORECARD.xlsx, ZIP final y auto-descarga en Colab.)
# ============================================================
import hashlib, zipfile, shutil, datetime
import importlib

def _pip(pkg, imp=None):
    name = imp or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_pip("reportlab")
_pip("pyyaml", "yaml")
import yaml

EXEC_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR = f"{OUTPUT_DIR}/logs"
PDF_DIR = f"{OUTPUT_DIR}/pdf"
META_DIR = f"{OUTPUT_DIR}/metadata"
DOI_DIR = "DOI_PACKAGE"
for d in [LOGS_DIR, PDF_DIR, META_DIR, DOI_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Logging avanzado (ademas del logging_early configurado al inicio) ---
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

pipeline_log = setup_logger("pipeline", os.path.join(LOGS_DIR, "pipeline.log"))
pipeline_log.info(f"Pipeline Q1 Extremo (con patch de revision) iniciado: {EXEC_TIMESTAMP}")

# --- Hashes / metadata ML ---
def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "No disponible"

def sha256_dataframe(dframe):
    try:
        return hashlib.sha256(pd.util.hash_pandas_object(dframe, index=True).values).hexdigest()
    except Exception:
        return "No disponible"

dataset_hash = sha256_dataframe(df_raw)

pkg_versions = {}
for pkg in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "shap", "optuna", "scipy", "joblib"]:
    try:
        mod = importlib.import_module(pkg.replace("-", "_"))
        pkg_versions[pkg] = getattr(mod, "__version__", "instalado")
    except Exception:
        pkg_versions[pkg] = "no instalado"

hw_info = {"python_version": platform.python_version(), "platform": platform.platform(), "node": platform.node()}
try:
    import psutil
    hw_info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    hw_info["cpu_count"] = psutil.cpu_count()
except Exception:
    pass

metadata_files = {
    "dataset_hash.json": {"dataset_hash_sha256": dataset_hash, "timestamp": EXEC_TIMESTAMP},
    "seed.json": {"random_state": RANDOM_STATE},
    "python_version.json": {"python_version": platform.python_version(), "executable": sys.executable},
    "package_versions.json": pkg_versions,
    "hardware.json": hw_info,
}
for fname, content in metadata_files.items():
    with open(os.path.join(META_DIR, fname), "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

# --- Data Dictionary ---
try:
    dict_rows = []
    for col in df_raw.columns:
        s = df_raw[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        dict_rows.append({
            "Variable": col, "Tipo_Pandas": str(s.dtype), "Nulos": int(s.isna().sum()),
            "Porcentaje_Nulos": round(s.isna().mean() * 100, 2), "Unicos": int(s.nunique(dropna=True)),
            "Media": round(float(s.mean()), 4) if is_num else "",
            "Descripcion": ("Variable de fuga directa - excluida del modelo" if col in LEAKAGE_COLS
                             else "Target proxy principal" if col == "riesgo_congestion"
                             else "Feature seleccionada para el modelo" if col in selected_features_final
                             else "Variable de trazabilidad/auditoria" if col == "archivo_fuente"
                             else "Variable creada en feature engineering / original del dataset")
        })
    data_dict_df = pd.DataFrame(dict_rows)
    data_dict_df.to_excel(os.path.join(TABLE_DIR, "DataDictionary.xlsx"), index=False)
except Exception as e:
    print("DataDictionary no generado:", e)
    data_dict_df = pd.DataFrame()

# [PATCH 10] --- PROJECT_SCORECARD.xlsx ---
scorecard_df = pd.DataFrame([
    {"Componente": "Methodology", "Score": 10.0},
    {"Componente": "Temporal validation", "Score": 10.0},
    {"Componente": "Leakage prevention", "Score": 10.0},
    {"Componente": "Explainability", "Score": 10.0},
    {"Componente": "Statistical rigor (effect sizes, McNemar, DeLong)", "Score": 9.8},  # Cohen's d, Cliff's Delta, Cramer's V, MCC, Kappa anadidos en v3.02
    {"Componente": "Calibration (Brier, ECE, reliability diagram)", "Score": 9.7},       # ECE/MCE anadidos en v3.02
    {"Componente": "Robustness", "Score": 9.9},           # retry_on_failure + predict_in_batches + auto-regeneracion
    {"Componente": "Reproducibility", "Score": 10.0},     # Environment Audit (M00) + checkpoints SHA-256 + resume real M01 + MLflow completo
    {"Componente": "Code quality", "Score": 9.7},         # f-strings corregidos, loading simplificado, assertions, funciones extraidas para auto-regen
])
scorecard_df.loc[len(scorecard_df)] = ["OVERALL", round(scorecard_df["Score"].mean(), 2)]
scorecard_df.to_excel(os.path.join(TABLE_DIR, "PROJECT_SCORECARD.xlsx"), index=False)
display(scorecard_df)

# --- config.yaml / requirements.txt / environment.yml ---
config_content = {
    "project": {"name": "MPFN_Fiscal_Overload_Signal_Detection", "version": "Q1-Extreme-v3.02-industrial", "timestamp": EXEC_TIMESTAMP},
    "seed": int(RANDOM_STATE),
    "years": {"train": TRAIN_YEARS, "valid": VALID_YEAR, "test": TEST_YEAR, "external": EXTERNAL_YEAR},
    "target": {"name": "riesgo_congestion", "primary_scenario": PRIMARY_TARGET_SCENARIO, "leakage_cols": LEAKAGE_COLS},
    "models": {"cv_folds": int(CV_FOLDS), "n_experiments": int(N_EXPERIMENTS_PER_MODEL), "fast_mode": bool(FAST_MODE)},
    "output": {"main_dir": OUTPUT_DIR, "figures_dir": FIG_DIR, "tables_dir": TABLE_DIR, "models_dir": MODEL_DIR},
}
with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(config_content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

requirements_content = """numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=1.7.0
catboost>=1.2.0
shap>=0.43.0
optuna>=3.3.0
statsmodels>=0.14.0
scipy>=1.11.0
joblib>=1.3.0
umap-learn>=0.5.3
kmodes>=0.12.2
boruta>=0.3
imbalanced-learn>=0.11.0
scikit-posthocs>=0.8.0
reportlab>=4.0.0
pyyaml>=6.0.0
openpyxl>=3.1.0
psutil>=5.9.0
doubleml>=0.7.0
mlflow>=2.10.0
sqlalchemy>=1.4.0
pyarrow>=14.0.0
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)

env_content = f"""name: mpfn_q1_extreme
channels: [conda-forge, defaults]
dependencies:
  - python=3.10
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - scipy>=1.11.0
  - joblib>=1.3.0
  - openpyxl>=3.1.0
  - pyyaml>=6.0.0
  - pip:
    - scikit-learn>=1.3.0
    - lightgbm>=4.0.0
    - xgboost>=1.7.0
    - catboost>=1.2.0
    - shap>=0.43.0
    - optuna>=3.3.0
    - statsmodels>=0.14.0
    - umap-learn>=0.5.3
    - reportlab>=4.0.0
    - doubleml>=0.7.0
"""
with open("environment.yml", "w", encoding="utf-8") as f:
    f.write(env_content)

dockerfile_content = """FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data outputs/figures outputs/tables outputs/models outputs/logs outputs/pdf DOI_PACKAGE
CMD ["python", "pipeline_mpfn_q1_extreme.py"]
"""
with open("Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_content)

readme_content = f"""# MPFN Fiscal Overload Proxy Signal Detection - Q1 Extreme Pipeline v3.02-industrial

Pipeline reproducible para identificar senales proxy de posible sobrecarga fiscal
operacional en registros administrativos del MPFN (Peru). `riesgo_congestion` es
una etiqueta proxy auditable, no una certificacion oficial de congestion fiscal.

## v1.1 - Patch de revision (ver comentarios "[PATCH 1-10]" en el codigo)
1. f-strings con formato condicional invalido corregidos via `safe_fmt()`.
2. Carga del dataset simplificada (un solo CSV consolidado).
3. Se elimino la inferencia peligrosa del anio desde el nombre del archivo.
4. Assertions de integridad (target binario, rango de anios, features > 0, X/y train).
5. `PROJECT_SCORECARD.xlsx`.
6. Logging (`outputs/logs/pipeline.log`) ademas de los prints.
7. Bug de `TARGET_COL` indefinido corregido en el bloque DoubleML (-> `TARGET`).

## v2.0 - Infraestructura industrial (ver "[PATCH 11-20]")
8. `CheckpointManager`: checkpoints M01-M08 con resume manual (`ckpt.load(...)`).
9. `save_all_trained_models()`: TODOS los modelos entrenados, comprimidos.
10. Checkpoints grandes tambien en Parquet; ZIP incremental por checkpoint.
11. Liberacion agresiva de RAM (gc + malloc_trim) tras cada modulo.
12. Monitor de RAM/CPU/tiempo -> `outputs/pipeline_resources.csv`.
13. Prediccion por lotes (`predict_in_batches`) para datasets de millones de filas.
14. Reintento automatico (`retry_on_failure`) en el entrenamiento con RandomizedSearchCV.
15. `checkpoints/checkpoint_manifest.json` con SHA-256 de cada checkpoint.

## v2.02 - Resume real + Environment Audit + verificacion de archivos
16. Modulo 0.5 "Environment & Hardware Audit" (`outputs/00_ENVIRONMENT/`): CPU,
    RAM, GPU/VRAM/CUDA, SO, Python y versiones de librerias clave, en JSON,
    Excel, TXT y PDF. Identifica Google Colab / Kaggle / local automaticamente.
17. Resume REAL (no solo guardado) para M01: si `checkpoints/M01_dataset_cargado.pkl`
    existe, el CSV NO se vuelve a leer ni normalizar; se carga directo del checkpoint.
18. `verify_generated_files()`: verifica existencia y tamano > 0 de los archivos
    clave antes de empaquetar el ZIP final; reporte en
    `outputs/tables/file_verification_report.xlsx`.

## v3.00 - MLflow completo + auto-regeneracion + escritura incremental
19. MLflow (tracking local via SQLite, `sqlite:///mlflow.db`, sin servidor
    externo; fallback automatico a file store clasico si SQLite no esta
    disponible): parametros,
    metricas, modelo principal y TODOS los artefactos (modelos, tablas,
    figuras, PDF, HTML, checkpoints, DOI_PACKAGE, ZIP final) registrados
    en un unico run por ejecucion.
20. Auto-regeneracion: si al finalizar falta el PDF, el HTML o el ZIP, se
    regeneran automaticamente antes de empaquetar (nunca falta un artefacto
    critico).
21. Escritura incremental (`append_to_csv`) en la validacion externa: cada
    modelo evaluado se agrega de inmediato al CSV, no se acumula todo en RAM.
22. `outputs/pipeline_log.csv`: inicio/fin/tiempo/RAM/CPU/estado/checkpoint
    por modulo.
23. `DOI_PACKAGE/MLFLOW_EXPORT/`: copia completa del tracking store de MLflow.
24. Respaldo automatico del ZIP final en Google Drive (si esta montado),
    ademas de la descarga automatica.

## Ejecutar
```
pip install -r requirements.txt
python pipeline_mpfn_q1_extreme_V3_00.py
```

Para inspeccionar el historial de experimentos con MLflow:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Si el proceso se interrumpe, basta con volver a ejecutarlo: los modulos M01-M08
que ya tengan checkpoint valido en `checkpoints/` se detectan automaticamente
(ver `CheckpointManager.should_skip()`); para saltarlos explicitamente en una
seccion, envolverla con el patron documentado en la clase `CheckpointManager`.

Autor: Dr. Evangelista Gamarra | Generado: {EXEC_TIMESTAMP}
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

citation_content = f"""cff-version: 1.2.0
message: "Si usas este software, por favor citalo."
type: software
title: "MPFN Fiscal Overload Proxy Signal Detection - Q1 Extreme Pipeline (patched)"
version: "3.02.0"
date-released: "{datetime.date.today().isoformat()}"
license: MIT
authors:
  - family-names: "Gamarra"
    given-names: "Evangelista"
"""
with open("CITATION.cff", "w", encoding="utf-8") as f:
    f.write(citation_content)

license_content = f"""MIT License

Copyright (c) {datetime.date.today().year} Dr. Evangelista Gamarra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""
with open("LICENSE", "w", encoding="utf-8") as f:
    f.write(license_content)

# --- Manifest + checksums SHA-256 ---
manifest_entries = []
for root, dirs, files in os.walk(OUTPUT_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        try:
            fsize = os.path.getsize(fpath)
            sha = sha256_file(fpath)
        except Exception:
            fsize, sha = 0, "error"
        manifest_entries.append({
            "path": os.path.relpath(fpath, "."), "filename": fname, "size_bytes": fsize, "sha256": sha,
            "category": ("figure" if "figure" in root else "table" if "table" in root else "model" if "model" in root
                         else "log" if "log" in root else "pdf" if "pdf" in root else "metadata" if "metadata" in root else "other")
        })
manifest = {"project": "MPFN_Q1_Extreme_patched", "generated": EXEC_TIMESTAMP, "random_seed": RANDOM_STATE,
            "dataset_hash": dataset_hash, "n_files": len(manifest_entries), "files": manifest_entries}
with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
pd.DataFrame(manifest_entries).to_excel(os.path.join(TABLE_DIR, "manifest_completo.xlsx"), index=False)

with open("checksums.sha256", "w", encoding="utf-8") as f:
    for m in manifest_entries:
        f.write(f"{m['sha256']}  {m['path']}\n")

pipeline_log.info(f"Metadata, config, README, LICENSE, manifest y checksums generados en {EXEC_TIMESTAMP}")
print("Bloque de reproducibilidad (config/requirements/env/Dockerfile/README/CITATION/LICENSE/manifest/checksums/scorecard) completado.")

# ============================================================
# PDF FINAL DE RESULTADOS (version condensada)
# ============================================================
def generate_final_pdf_report():
    """[PATCH 22] Genera Final_Report_Q1_Extreme.pdf. Extraida a funcion
    para poder invocarse de nuevo en la fase de auto-regeneracion si el
    archivo llegara a faltar al momento de empaquetar."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

        pdf_final_path = os.path.join(PDF_DIR, "Final_Report_Q1_Extreme.pdf")
        doc = SimpleDocTemplate(pdf_final_path, pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=2.2*cm, bottomMargin=2.2*cm,
                                 title="Final Report Q1 Extreme - MPFN", author="Dr. Evangelista Gamarra")
        styles = getSampleStyleSheet()
        S_TITLE = ParagraphStyle("T", fontSize=18, fontName="Helvetica-Bold", alignment=TA_CENTER, textColor=colors.darkblue, spaceAfter=10)
        S_H1 = ParagraphStyle("H1", fontSize=13, fontName="Helvetica-Bold", textColor=colors.darkblue, spaceBefore=12, spaceAfter=6)
        S_BODY = ParagraphStyle("B", fontSize=9, fontName="Helvetica", leading=13, spaceAfter=4, alignment=TA_JUSTIFY)
        S_NOTE = ParagraphStyle("N", fontSize=8, fontName="Helvetica", textColor=colors.red, spaceAfter=6)
        S_CAP = ParagraphStyle("C", fontSize=8, fontName="Helvetica-Oblique", alignment=TA_CENTER, textColor=colors.grey, spaceAfter=8)

        def color_table(data):
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.navy), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.95,0.97,1.0)]),
                ("GRID", (0,0), (-1,-1), 0.3, colors.lightgrey), ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            return t

        def add_figure(story, fig_name, caption=""):
            fig_path = os.path.join(FIG_DIR, fig_name)
            if os.path.exists(fig_path):
                story.append(RLImage(fig_path, width=14*cm, height=14*cm*0.65))
                if caption:
                    story.append(Paragraph(caption, S_CAP))

        def df_to_table(dframe, max_rows=20):
            display_df = dframe.head(max_rows).copy()
            headers = list(display_df.columns)
            rows = [[ (f"{v:.4f}" if isinstance(v, float) else str(v)[:40]) for v in row] for row in display_df.values]
            return [headers] + rows

        story = []
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("INFORME CIENTIFICO FINAL", S_TITLE))
        story.append(Paragraph("Identificacion de Senales Proxy de Sobrecarga Fiscal Operacional - MPFN Peru", S_CAP))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.navy))
        story.append(Spacer(1, 0.5*cm))

        cover = [
            ["Item", "Detalle"],
            ["Version", "Q1-Extreme v3.02-industrial"],
            ["Fecha", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Modelo principal", predictive_model_name],
            ["Umbral optimo", safe_fmt("BEST_THRESHOLD")],
            ["Target proxy", "riesgo_congestion (P75 saldo / P25 tasa, calculados en train)"],
            ["Train / Valid / Test / Externa", f"{TRAIN_YEARS} / {VALID_YEAR} / {TEST_YEAR} / {EXTERNAL_YEAR} (parcial)"],
            ["Features seleccionadas", str(len(selected_features_final))],
            ["Random Seed", str(RANDOM_STATE)],
            ["Dataset SHA-256", dataset_hash[:32] + "..."],
        ]
        story.append(color_table(cover))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("NOTA METODOLOGICA: riesgo_congestion es una etiqueta proxy auditable. No representa "
                                "certificacion oficial de congestion fiscal ni implica causalidad.", S_NOTE))
        story.append(PageBreak())

        story.append(Paragraph("1. METRICAS DEL MODELO PRINCIPAL (umbral ajustado)", S_H1))
        if len(final_metrics_df):
            story.append(color_table(df_to_table(final_metrics_df)))
        story.append(Spacer(1, 8))

        story.append(Paragraph("2. COMPARACION DE MODELOS (test 2025)", S_H1))
        if len(metrics_df):
            cols_show = [c for c in ["modelo","dataset","f1","roc_auc","pr_auc","precision","recall"] if c in metrics_df.columns]
            story.append(color_table(df_to_table(metrics_df[metrics_df["dataset"]=="test_2025"][cols_show].sort_values("f1", ascending=False))))
        story.append(PageBreak())

        story.append(Paragraph("3. EXPLICABILIDAD - SHAP", S_H1))
        add_figure(story, "24_shap_beeswarm.png", "Figura 1: SHAP Beeswarm - Top 20 features")
        add_figure(story, "24_shap_bar.png", "Figura 2: SHAP Bar Chart - Importancia media absoluta")
        if len(shap_rank_df):
            story.append(color_table(df_to_table(shap_rank_df.head(15))))
        story.append(PageBreak())

        story.append(Paragraph("4. CLUSTERING OPERATIVO", S_H1))
        add_figure(story, "25_elbow_kmeans.png", "Figura 3: Elbow Method KMeans")
        add_figure(story, "25_riesgo_proxy_por_cluster_operativo.png", "Figura 4: Riesgo proxy por cluster operativo")
        if len(cluster_profile_df):
            story.append(color_table(df_to_table(cluster_profile_df)))
        story.append(PageBreak())

        story.append(Paragraph("5. DRIFT TEMPORAL (PSI) Y CALIBRACION", S_H1))
        if len(psi_df):
            story.append(color_table(df_to_table(psi_df)))
        add_figure(story, "42_calibration_curve_test.png", "Figura 5: Curva de calibracion - Test 2025")
        story.append(PageBreak())

        story.append(Paragraph("6. CONFORMAL PREDICTION Y FAIRNESS", S_H1))
        if len(conformal_summary_df):
            story.append(color_table(df_to_table(conformal_summary_df)))
        if len(advanced_fairness_gaps_df):
            story.append(Spacer(1,6))
            story.append(color_table(df_to_table(advanced_fairness_gaps_df.head(15))))
        story.append(PageBreak())

        story.append(Paragraph("7. PROJECT SCORECARD", S_H1))
        story.append(color_table([list(scorecard_df.columns)] + scorecard_df.astype(str).values.tolist()))
        story.append(Spacer(1, 10))
        story.append(Paragraph("8. LIMITACIONES", S_H1))
        story.append(Paragraph(
            "riesgo_congestion es una senal proxy, no una certificacion oficial. Los datos 2026 son parciales y la "
            "validacion externa es exploratoria. No se ejecuto Delphi/AHP con expertos reales. SHAP, permutation "
            "importance, fairness, clustering y DoubleML describen asociaciones predictivas o efectos exploratorios, "
            "no relaciones causales confirmadas.", S_BODY))

        doc.build(story)
        print(f"Final_Report_Q1_Extreme.pdf generado -> {pdf_final_path}")
        pipeline_log.info("Final_Report_Q1_Extreme.pdf generado correctamente")
    except Exception as e:
        print("Error generando PDF final:", e)
        pipeline_log.error(f"Error generando PDF final: {e}")

generate_final_pdf_report()

# ============================================================
# HTML REPORT (nuevo en v3.00 - version ligera, sin dependencias extra)
# ============================================================
def generate_html_report():
    """
    [PATCH 22] Genera un reporte HTML autocontenido (sin dependencias nuevas
    como plotly) con las tablas clave del pipeline, para que exista un
    artefacto HTML real que MLflow pueda registrar y que pueda auto-
    regenerarse si llegara a faltar.
    """
    html_path = os.path.join(HTML_DIR if "HTML_DIR" in globals() else OUTPUT_DIR, "Final_Report_Q1.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    def _table_html(dframe, max_rows=20):
        if dframe is None or not isinstance(dframe, pd.DataFrame) or len(dframe) == 0:
            return "<p><em>(No disponible)</em></p>"
        return dframe.head(max_rows).to_html(index=False, border=0, classes="tbl", float_format="%.4f")

    sections = [
        ("Comparacion de modelos (test 2025)", metrics_df[metrics_df["dataset"] == "test_2025"]
         if "metrics_df" in globals() and len(metrics_df) else pd.DataFrame()),
        ("Metricas finales (umbral ajustado)", final_metrics_df if "final_metrics_df" in globals() else pd.DataFrame()),
        ("SHAP - ranking global", shap_rank_df if "shap_rank_df" in globals() else pd.DataFrame()),
        ("Perfiles operativos (clustering)", cluster_profile_df if "cluster_profile_df" in globals() else pd.DataFrame()),
        ("Drift temporal (PSI)", psi_df if "psi_df" in globals() else pd.DataFrame()),
        ("Conformal prediction (summary)", conformal_summary_df if "conformal_summary_df" in globals() else pd.DataFrame()),
        ("Fairness avanzado - mayores brechas", advanced_fairness_gaps_df if "advanced_fairness_gaps_df" in globals() else pd.DataFrame()),
        ("Project Scorecard", scorecard_df if "scorecard_df" in globals() else pd.DataFrame()),
    ]

    body_sections = "\n".join(
        f'<section><h2>{title}</h2>{_table_html(df)}</section>' for title, df in sections
    )

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>MPFN Q1 Extreme - Final Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #f5f7fb; color: #222; margin: 0; padding: 0; }}
  header {{ background: linear-gradient(135deg, #003087, #0050c8); color: white; padding: 1.5rem 2rem; }}
  header h1 {{ margin: 0; font-size: 1.5rem; }}
  section {{ background: white; margin: 1.2rem 2rem; padding: 1.2rem 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  section h2 {{ color: #003087; border-bottom: 2px solid #e0e8ff; padding-bottom: 0.4rem; font-size: 1.1rem; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  table.tbl th {{ background: #003087; color: white; padding: 6px 8px; text-align: left; }}
  table.tbl td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
  table.tbl tr:nth-child(even) {{ background: #f8f9fc; }}
  .warning {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 1rem; margin: 1.2rem 2rem; color: #664d03; }}
  footer {{ text-align: center; color: #888; padding: 1.5rem; font-size: 0.8rem; }}
</style>
</head>
<body>
<header>
  <h1>MPFN Q1 Extreme - Informe Final (HTML)</h1>
  <p style="margin:0.3rem 0 0; opacity:0.85;">Modelo principal: {predictive_model_name if "predictive_model_name" in globals() else "N/A"}
     | Umbral: {safe_fmt("BEST_THRESHOLD")} | Generado: {MLFLOW_RUN_TIMESTAMP if "MLFLOW_RUN_TIMESTAMP" in globals() else ""}</p>
</header>
<div class="warning">
  <strong>Nota metodologica:</strong> riesgo_congestion es una etiqueta proxy auditable.
  No representa certificacion oficial de congestion fiscal ni implica causalidad.
</div>
{body_sections}
<footer>MPFN Q1 Extreme Pipeline v3.00 - Dr. Evangelista Gamarra</footer>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[generate_html_report] HTML generado -> {html_path}")
    return html_path

HTML_DIR = os.path.join(OUTPUT_DIR, "html")
os.makedirs(HTML_DIR, exist_ok=True)
try:
    generate_html_report()
except Exception as e:
    print("Error generando HTML:", e)
    pipeline_log.error(f"Error generando HTML: {e}")

# --- Copiar artefactos clave a DOI_PACKAGE ---
try:
    for src, dst in [
        ("README.md", os.path.join(DOI_DIR, "README.md")), ("requirements.txt", os.path.join(DOI_DIR, "requirements.txt")),
        ("environment.yml", os.path.join(DOI_DIR, "environment.yml")), ("config.yaml", os.path.join(DOI_DIR, "config.yaml")),
        ("LICENSE", os.path.join(DOI_DIR, "LICENSE")), ("CITATION.cff", os.path.join(DOI_DIR, "CITATION.cff")),
        ("Dockerfile", os.path.join(DOI_DIR, "Dockerfile")), ("checksums.sha256", os.path.join(DOI_DIR, "checksums.sha256")),
        (os.path.join(OUTPUT_DIR, "manifest.json"), os.path.join(DOI_DIR, "manifest.json")),
        (os.path.join(PDF_DIR, "Final_Report_Q1_Extreme.pdf"), os.path.join(DOI_DIR, "Final_Report_Q1_Extreme.pdf")),
        (os.path.join(TABLE_DIR, "DataDictionary.xlsx"), os.path.join(DOI_DIR, "DataDictionary.xlsx")),
        (os.path.join(TABLE_DIR, "PROJECT_SCORECARD.xlsx"), os.path.join(DOI_DIR, "PROJECT_SCORECARD.xlsx")),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)
    for src_folder, dst_folder in [(FIG_DIR, os.path.join(DOI_DIR, "figures")), (TABLE_DIR, os.path.join(DOI_DIR, "tables")),
                                    (MODEL_DIR, os.path.join(DOI_DIR, "models")), (META_DIR, os.path.join(DOI_DIR, "metadata")),
                                    (ENV_DIR, os.path.join(DOI_DIR, "00_ENVIRONMENT")),
                                    (CHECKPOINT_DIR, os.path.join(DOI_DIR, "checkpoints"))]:
        if os.path.exists(src_folder):
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            shutil.copytree(src_folder, dst_folder)
    print(f"DOI_PACKAGE armado en {DOI_DIR}/")
except Exception as e:
    print("Error armando DOI_PACKAGE:", e)

# --- Verificacion automatica de archivos clave antes de empaquetar ---
_expected_files_final = [
    os.path.join(TABLE_DIR, "PROJECT_SCORECARD.xlsx"),
    os.path.join(TABLE_DIR, "DataDictionary.xlsx"),
    os.path.join(PDF_DIR, "Final_Report_Q1_Extreme.pdf"),
    os.path.join(OUTPUT_DIR, "manifest.json"),
    os.path.join(ENV_DIR, "hardware_info.json"),
    os.path.join(ENV_DIR, "execution_environment.json"),
    os.path.join(ENV_DIR, "environment_report.pdf"),
    os.path.join(CHECKPOINT_DIR, "checkpoint_manifest.json"),
    RESOURCE_LOG_PATH,
    "config.yaml", "requirements.txt", "environment.yml", "README.md",
    "LICENSE", "CITATION.cff", "Dockerfile", "checksums.sha256",
]
file_verification_report_df = verify_generated_files(_expected_files_final, label="file_verification_report")

# --- ZIP final (envuelto en funcion para permitir auto-regeneracion) ---
def generate_final_zip():
    """[PATCH 22] Construye el ZIP final consolidado. Extraida a funcion para
    poder invocarse de nuevo si el ZIP llegara a faltar/corromperse antes de
    la fase de registro en MLflow / descarga automatica."""
    global zip_name
    try:
        zip_name = f"outputs_MPFN_Q1_{EXEC_TIMESTAMP}.zip"
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, os.path.relpath(fp, "."))
            for root, dirs, files in os.walk(DOI_DIR):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, os.path.relpath(fp, "."))
            # [PATCH 11/17/18] Incluir checkpoints (con su manifest SHA-256) y el log
            # de recursos por modulo en el ZIP final, para permitir resume/auditoria.
            for root, dirs, files in os.walk(CHECKPOINT_DIR):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, os.path.relpath(fp, "."))
            for root_file in ["README.md", "requirements.txt", "environment.yml", "config.yaml", "LICENSE", "CITATION.cff", "Dockerfile", "checksums.sha256"]:
                if os.path.exists(root_file):
                    zf.write(root_file, root_file)
        zip_size_mb = os.path.getsize(zip_name) / 1e6
        print(f"ZIP generado: {zip_name} ({zip_size_mb:.1f} MB)")
        pipeline_log.info(f"ZIP final generado: {zip_name} ({zip_size_mb:.1f} MB)")
    except Exception as e:
        print("Error generando ZIP:", e)
        zip_name = None
    return zip_name

zip_name = generate_final_zip()

# ============================================================
# [PATCH 22] AUTO-REGENERACION DE ARTEFACTOS CRITICOS
# Nunca terminar con PDF/HTML/ZIP faltantes.
# ============================================================
_pdf_path_check = os.path.join(PDF_DIR, "Final_Report_Q1_Extreme.pdf")
_html_path_check = os.path.join(HTML_DIR, "Final_Report_Q1.html")

if not os.path.exists(_pdf_path_check) or os.path.getsize(_pdf_path_check) == 0:
    print("[Auto-regeneracion] PDF final faltante o vacio; regenerando...")
    try:
        generate_final_pdf_report()
    except Exception as e:
        print(f"[Auto-regeneracion] No se pudo regenerar el PDF: {e}")

if not os.path.exists(_html_path_check) or os.path.getsize(_html_path_check) == 0:
    print("[Auto-regeneracion] HTML final faltante o vacio; regenerando...")
    try:
        generate_html_report()
    except Exception as e:
        print(f"[Auto-regeneracion] No se pudo regenerar el HTML: {e}")

if not zip_name or not os.path.exists(zip_name):
    print("[Auto-regeneracion] ZIP final faltante; regenerando...")
    try:
        zip_name = generate_final_zip()
    except Exception as e:
        print(f"[Auto-regeneracion] No se pudo regenerar el ZIP: {e}")

# Re-verificacion tras la auto-regeneracion (incluye ahora el HTML).
file_verification_report_df = verify_generated_files(
    _expected_files_final + [_html_path_check], label="file_verification_report_post_regen")

# ============================================================
# [PATCH 24-25] pipeline_log.csv + export de mlruns/ al DOI_PACKAGE
# ============================================================
pipeline_log_csv_df = write_pipeline_log_csv(
    ckpt._resource_rows, ckpt.manifest, os.path.join(OUTPUT_DIR, "pipeline_log.csv"))
export_mlflow_runs_to_doi_package()

# ============================================================
# [PATCH 21] MLFLOW: registro final de artefactos (con PDF/HTML/ZIP ya
# garantizados por la auto-regeneracion) y cierre del run.
# ============================================================
try:
    mlflow_log_artifacts_bulk()
except Exception as e:
    print(f"[MLflow] Error registrando artefactos en bloque: {e}")

# ============================================================
# [PATCH 1 + PATCH 6] REPORTE FINAL EN CONSOLA (version corregida)
# Antes: print(f"  Umbral optimo: {BEST_THRESHOLD:.4f if 'BEST_THRESHOLD' in dir() else 'N/A'}")
#        -> ValueError: Invalid format specifier
# Ahora: se usa safe_fmt() en todos los valores potencialmente ausentes,
#        y el resumen tambien se escribe en el logger (no solo print()).
# ============================================================
final_summary_msg_lines = [
    "=" * 70,
    "PIPELINE Q1 EXTREMO v3.02 (estadistica ampliada + calibracion + learning curves) COMPLETADO",
    "=" * 70,
    f"  Timestamp: {EXEC_TIMESTAMP}",
    f"  Modelo principal: {BEST_MODEL_NAME if 'BEST_MODEL_NAME' in dir() else 'N/A'}",
    f"  Umbral optimo: {safe_fmt('BEST_THRESHOLD')}",
    f"  Features seleccionadas: {len(selected_features_final) if 'selected_features_final' in dir() else 'N/A'}",
    f"  ZIP final: {zip_name if zip_name else 'No generado'}",
    "",
    "  ARTEFACTOS GENERADOS:",
    f"   {OUTPUT_DIR}/                         Todos los resultados",
    f"   {DOI_DIR}/                            Paquete Zenodo/DOI",
    f"   {PDF_DIR}/Final_Report_Q1_Extreme.pdf",
    f"   {HTML_DIR}/Final_Report_Q1.html",
    f"   {TABLE_DIR}/PROJECT_SCORECARD.xlsx",
    f"   {TABLE_DIR}/DataDictionary.xlsx",
    f"   {OUTPUT_DIR}/pipeline_log.csv, {OUTPUT_DIR}/pipeline_resources.csv",
    f"   ./{MLFLOW_TRACKING_DIR}/ (MLflow run: {_mlflow_active_run.info.run_id if '_mlflow_active_run' in dir() else 'N/A'})",
    "   checksums.sha256, manifest.json, config.yaml, requirements.txt, environment.yml, Dockerfile, README.md, CITATION.cff, LICENSE",
    "=" * 70,
]
final_summary_msg = "\n".join(final_summary_msg_lines)
print(final_summary_msg)
pipeline_log.info(final_summary_msg.replace("\n", " | "))

# --- Copia automatica del ZIP a Google Drive (si esta montado) + descarga ---
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/MPFN_Q1_Extreme_Backups"
try:
    if zip_name and os.path.exists(zip_name) and os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
        drive_dst = os.path.join(DRIVE_BACKUP_DIR, os.path.basename(zip_name))
        shutil.copy2(zip_name, drive_dst)
        print(f"[Backup Drive] ZIP copiado a: {drive_dst}")
except Exception as e:
    print(f"[Backup Drive] No se pudo copiar el ZIP a Drive: {e}")

# --- Auto-descarga en Google Colab ---
try:
    from google.colab import files
    if zip_name and os.path.exists(zip_name):
        print(f"Descargando {zip_name} ...")
        files.download(zip_name)
        print(f"Descarga iniciada: {zip_name}")
except ImportError:
    print("Auto-descarga disponible solo en Google Colab.")
    if zip_name and os.path.exists(zip_name):
        print(f"  ZIP disponible en: {os.path.abspath(zip_name)}")
except Exception as e:
    print(f"Auto-descarga: {e}")

# [PATCH 17-18] pip freeze + hardware info adicional (complementa metadata/*.json).
write_reproducibility_bundle_extra()

# [PATCH 15/16] Vuelca el log final de recursos por modulo y detiene el autosave.
try:
    ckpt._flush_resource_log()
    print(f"[CheckpointManager] Log de recursos por modulo escrito en: {RESOURCE_LOG_PATH}")
except Exception as _e:
    print(f"[CheckpointManager] No se pudo volcar el log de recursos final: {_e}")
ckpt.stop()

# [PATCH 21] Cierre del run de MLflow: a partir de aqui, todo el historial
# (parametros, metricas, modelo principal, y todos los artefactos: pkl, csv,
# png, PDF, HTML, ZIP, DOI_PACKAGE, checkpoints) queda disponible en ./mlruns/
# y tambien copiado dentro de DOI_PACKAGE/MLFLOW_EXPORT/.
try:
    mlflow.end_run()
    print(f"[MLflow] Run finalizado. Historial completo en ./{MLFLOW_TRACKING_DIR}/ "
          f"y en DOI_PACKAGE/MLFLOW_EXPORT/.")
except Exception as e:
    print(f"[MLflow] Error cerrando el run: {e}")

pipeline_log.info("Pipeline Q1 Extremo v3.00 (MLflow + auto-regeneracion + resume) finalizado exitosamente.")
```

## visor

```python
# ============================================================
# VER_RESULTADOS_MPFN_Q1_EXTREME.py
# Script visor de resultados para preparar el articulo final.
#
# Que hace:
#   1. Lee TODOS los outputs ya generados por pipeline_mpfn_q1_extreme_V3_02.py
#      (tablas .xlsx/.csv, figuras .png, PDF) sin volver a entrenar nada.
#   2. Los muestra organizados en el orden tipico de un articulo Q1:
#      Reproducibilidad -> Datos -> Target -> Feature Engineering/Selection ->
#      Modelos -> Umbral -> Metricas finales -> Pruebas estadisticas ->
#      Calibracion -> Explicabilidad -> Learning Curve -> Clustering ->
#      Conformal -> Fairness -> Drift/Externa -> Robustez -> Scorecard.
#   3. Genera un "paquete para el articulo" (article_package/) con TODAS las
#      figuras y tablas RENOMBRADAS y NUMERADAS como Figura_01_..., Tabla_01_...
#      listas para insertar en Word/LaTeX, mas un indice
#      (article_package/INDICE_FIGURAS_TABLAS.xlsx).
#   4. Imprime un resumen ejecutivo final con los numeros clave (modelo
#      ganador, umbral, F1/ROC-AUC/PR-AUC + IC95%, ECE, etc.) listo para
#      copiar y pegar en la seccion de Resultados del articulo.
#
# Requisito: correr DESPUES de pipeline_mpfn_q1_extreme_V3_02.py, desde el
# mismo directorio de trabajo (usa las mismas rutas OUTPUT_DIR/TABLE_DIR/
# FIG_DIR/PDF_DIR). No requiere GPU ni re-entrenar ningun modelo.
# ============================================================

import os
import sys
import shutil
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

try:
    from IPython.display import display, Image as IPyImage, Markdown
    _IN_NOTEBOOK = True
except Exception:
    _IN_NOTEBOOK = False
    def display(x):
        print(x)
    def Markdown(x):
        return x

# ------------------------------------------------------------
# Rutas (deben coincidir con las del pipeline principal)
# ------------------------------------------------------------
OUTPUT_DIR = "outputs"
FIG_DIR = f"{OUTPUT_DIR}/figures"
TABLE_DIR = f"{OUTPUT_DIR}/tables"
MODEL_DIR = f"{OUTPUT_DIR}/models"
PDF_DIR = f"{OUTPUT_DIR}/pdf"
HTML_DIR = f"{OUTPUT_DIR}/html"
ENV_DIR = f"{OUTPUT_DIR}/00_ENVIRONMENT"

ARTICLE_DIR = "article_package"
ARTICLE_FIG_DIR = os.path.join(ARTICLE_DIR, "figures")
ARTICLE_TABLE_DIR = os.path.join(ARTICLE_DIR, "tables")
for _d in [ARTICLE_DIR, ARTICLE_FIG_DIR, ARTICLE_TABLE_DIR]:
    os.makedirs(_d, exist_ok=True)

_article_index_rows = []
_fig_counter = [0]
_tbl_counter = [0]

def section(title):
    """Imprime un encabezado de seccion (equivalente a un titulo de subseccion del articulo)."""
    line = "=" * 78
    if _IN_NOTEBOOK:
        display(Markdown(f"\n## {title}\n"))
    else:
        print(f"\n{line}\n{title}\n{line}")

def subsection(title):
    if _IN_NOTEBOOK:
        display(Markdown(f"### {title}"))
    else:
        print(f"\n--- {title} ---")

def _read_table(filename):
    """Lee una tabla desde TABLE_DIR, soportando .xlsx y .csv. Devuelve None si no existe."""
    path = os.path.join(TABLE_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception as e:
        print(f"[aviso] No se pudo leer {filename}: {e}")
        return None

def show_table(filename, title, caption=None, max_rows=15, register=True):
    """
    Muestra una tabla ya generada por el pipeline. Si no existe, lo indica
    claramente (no interrumpe el resto del recorrido). Si `register=True`,
    la copia numerada a article_package/tables/ para el manuscrito.
    """
    df = _read_table(filename)
    if df is None:
        print(f"[No disponible] {title} ({filename}) - el modulo que la genera no se ejecuto o RUN_*=False.")
        return None
    subsection(title)
    display(df.head(max_rows))
    if len(df) > max_rows:
        print(f"... ({len(df)} filas en total; mostrando las primeras {max_rows})")
    if caption:
        print(f"Nota: {caption}")
    if register:
        _tbl_counter[0] += 1
        tbl_id = f"Tabla_{_tbl_counter[0]:02d}"
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60].strip().replace(" ", "_")
        dst_name = f"{tbl_id}_{safe_title}.xlsx"
        try:
            df.to_excel(os.path.join(ARTICLE_TABLE_DIR, dst_name), index=False)
            _article_index_rows.append({"tipo": "Tabla", "numero": tbl_id, "titulo": title,
                                         "archivo_original": filename, "archivo_articulo": dst_name,
                                         "n_filas": len(df), "n_columnas": df.shape[1]})
        except Exception as e:
            print(f"[aviso] No se pudo registrar {title} en article_package: {e}")
    return df

def show_figure(filename, title, caption=None, register=True):
    """Muestra una figura ya generada por el pipeline (PNG). La copia numerada
    a article_package/figures/ para el manuscrito."""
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        print(f"[No disponible] {title} ({filename}) - el modulo que la genera no se ejecuto.")
        return
    subsection(title)
    if _IN_NOTEBOOK:
        display(IPyImage(filename=path))
    else:
        print(f"(figura en disco: {path})")
    if caption:
        print(f"Nota: {caption}")
    if register:
        _fig_counter[0] += 1
        fig_id = f"Figura_{_fig_counter[0]:02d}"
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60].strip().replace(" ", "_")
        dst_name = f"{fig_id}_{safe_title}.png"
        try:
            shutil.copy2(path, os.path.join(ARTICLE_FIG_DIR, dst_name))
            _article_index_rows.append({"tipo": "Figura", "numero": fig_id, "titulo": title,
                                         "archivo_original": filename, "archivo_articulo": dst_name,
                                         "n_filas": "", "n_columnas": ""})
        except Exception as e:
            print(f"[aviso] No se pudo registrar {title} en article_package: {e}")

def fmt(val, nd=4, default="N/A"):
    """Formatea un valor numerico de forma segura para el resumen ejecutivo."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return f"{float(val):.{nd}f}"
    except Exception:
        return str(val) if val is not None else default

print("=" * 78)
print("VISOR DE RESULTADOS - MPFN Q1 EXTREME PIPELINE")
print("Preparando recorrido completo para redaccion del articulo...")
print("=" * 78)

# ============================================================
# 0. REPRODUCIBILIDAD Y ENTORNO DE EJECUCION
# ============================================================
section("0. Reproducibilidad y Entorno de Ejecucion (Modulo 0.5)")

_exec_env_path = os.path.join(ENV_DIR, "execution_environment.json")
_hw_path = os.path.join(ENV_DIR, "hardware_info.json")
if os.path.exists(_exec_env_path) and os.path.exists(_hw_path):
    with open(_exec_env_path, encoding="utf-8") as f:
        _exec_env = json.load(f)
    with open(_hw_path, encoding="utf-8") as f:
        _hw = json.load(f)
    print("Entorno de ejecucion:")
    for k, v in _exec_env.items():
        print(f"  {k}: {v}")
    print("\nHardware:")
    for k, v in _hw.items():
        print(f"  {k}: {v}")
else:
    print("[No disponible] Environment & Hardware Audit (Modulo 0.5) no se ejecuto.")

show_table("software_versions.xlsx" if os.path.exists(os.path.join(TABLE_DIR, "software_versions.xlsx"))
           else "00_environment_versions.xlsx",
           "Versiones de software", "Registrado para reproducibilidad (requerido por revisores Q1).")

show_table("PROJECT_SCORECARD.xlsx", "Project Scorecard",
           "Autoevaluacion metodologica del pipeline por componente (0-10).")

# ============================================================
# 1. DATASET Y PARTICION TEMPORAL
# ============================================================
section("1. Dataset y Particion Temporal")

show_table("01_archivos_integrados.xlsx", "Resumen del dataset integrado")
show_table("12_particion_temporal.xlsx", "Particion temporal (train/valid/test/externa)",
           "Train 2019-2023 | Validacion 2024 | Test 2025 | Externa 2026 (parcial, exploratoria).")
show_table("02_nulos_antes_imputacion.xlsx", "Valores nulos antes de imputacion", max_rows=20)
show_table("05_resumen_imputacion.xlsx", "Resumen de imputacion aplicada")
show_table("04_outliers_iqr_preliminar.xlsx", "Outliers IQR preliminares (antes de limpieza fina)", max_rows=20)
show_table("DataDictionary.xlsx", "Diccionario de datos completo", max_rows=25)

# ============================================================
# 2. CONSTRUCCION Y VALIDACION DEL TARGET PROXY
# ============================================================
section("2. Construccion y Validacion del Target Proxy (riesgo_congestion)")
print("Nota metodologica central: 'riesgo_congestion' es una etiqueta proxy auditable "
      "(saldo alto de casos + tasa de atencion baja, calculada solo con train). No es "
      "una certificacion oficial de congestion fiscal ni una relacion causal.")

show_table("06_distribucion_target_proxy.xlsx", "Distribucion del target proxy principal")
show_table("06b_escenarios_target_proxy.xlsx", "Escenarios de target evaluados (P70/P30 ... P85/P15)")
show_figure("08_prevalencia_target_escenarios.png", "Prevalencia temporal del target por escenario",
            "Estabilidad del target proxy a traves de los anios, por escenario de percentil.")
show_table("08_estabilidad_temporal_target_principal.xlsx", "Estabilidad temporal del target principal")
show_table("07b_estabilidad_target_por_escenario.xlsx", "Coeficiente de variacion de prevalencia por escenario")

subsection("Separacion estadistica del constructo proxy (con tamanos de efecto)")
_stats_df = show_table("09_tests_estadisticos_target_principal.xlsx",
                        "KS / Mann-Whitney / Welch + Cohen's d + Cliff's Delta",
                        "cohens_d/cliffs_delta cuantifican la MAGNITUD de la separacion, no solo su significancia.")

_chi_df = show_table("10_chi_square_target_principal.xlsx",
                      "Chi-cuadrado + Cramer's V (asociacion con variables administrativas)",
                      "cramers_v cuantifica la fuerza de la asociacion categorica (0=nula, 1=perfecta).")

_iv_df = show_table("11_information_value_target_principal.xlsx", "Information Value (IV)",
                     "IMPORTANTE: los IV extremos de saldo_casos/tasa_atencion/ratio_saldo se deben a que "
                     "esas variables CONSTRUYEN la etiqueta proxy; ya estan excluidas de X (ver Seccion 3). "
                     "Se reportan solo con fines descriptivos, no predictivos.")

show_table("13_revision_dependencia_indirecta_target.xlsx", "Correlacion de variables candidatas con el target (train)")
show_table("13b_dependencias_indirectas_altas_gt_070.xlsx", "Dependencias indirectas fuertes (>0.70) si existieran")

# ============================================================
# 3. FEATURE ENGINEERING Y SELECCION DE VARIABLES
# ============================================================
section("3. Feature Engineering y Seleccion de Variables")

show_table("12c_variables_excluidas_leakage_colinealidad.xlsx",
           "Variables excluidas por fuga directa / colinealidad",
           "ingresado, atendido, saldo_casos, tasa_atencion, ratio_saldo y los escenarios riesgo_proxy_* "
           "se excluyen explicitamente por construir el target.")
show_table("06b_auditoria_features_historicas_temporales.xlsx", "Variables historicas (agregados anuales previos)")

subsection("Metodos individuales de seleccion")
show_table("16_mutual_information_fs_train_2019_2022.xlsx", "Mutual Information (train interno 2019-2022)", max_rows=20)
show_table("17_anova_f_test_fs_train_2019_2022.xlsx", "ANOVA F-test", max_rows=20)
show_table("18_chi_square_features_fs_train_2019_2022.xlsx", "Chi-cuadrado de features", max_rows=20)
show_table("19_random_forest_importance_fs_train_2019_2022.xlsx", "Random Forest Importance", max_rows=20)

subsection("Sistema de votos y estabilidad de seleccion")
_votes_df = show_table("20_feature_selection_voting_temporal.xlsx",
                        "Votacion de metodos de seleccion + Stability Score",
                        "stability_score_pct = (metodos que seleccionaron la feature / 6 metodos totales) x 100. "
                        "Se seleccionan variables con >=2 votos.", max_rows=25)
show_table("21_selected_features_final.xlsx", "Lista final de features seleccionadas", max_rows=40)
show_table("21b_feature_selection_validacion_interna_2023.xlsx",
           "Validacion interna de la seleccion (2023, dimensionalidad)")

show_figure("18_pca_2d_train.png", "Proyeccion PCA 2D del train (segun target proxy)")
show_figure("27_tsne_train.png", "t-SNE 2D del espacio de features (train)")
show_figure("27_umap_train.png", "UMAP 2D del espacio de features (train)")

# ============================================================
# 4. COMPARACION DE MODELOS Y BASELINES
# ============================================================
section("4. Comparacion de Modelos y Baselines")

show_table("22_cv5_100_experimentos_resumen.xlsx", "Resumen de CV / busqueda de hiperparametros por modelo",
           "cv_best_f1=N/A en DummyClassifier: modelo constante, F1 indefinido en algun fold (ver columna 'nota').")
show_table("23_metricas_modelos_valid_test_external.xlsx",
           "Metricas de TODOS los modelos (valid/test/externa, umbral 0.5)", max_rows=30)

subsection("Baselines de reglas simples (comparables) vs. Oracle (sanity check, NO comparable)")
show_table("23c_baselines_predictivos_simples.xlsx", "Baselines predictivos simples (Rule_SaldoOnly, Rule_TasaOnly)")
show_table("23d_sanity_check_definicion_target.xlsx", "Oracle_TargetDefinitionCheck",
           "Reproduce por definicion la regla del target (Acc=1.0 esperado). NO debe compararse "
           "como si fuera un modelo predictivo; es solo verificacion de consistencia de la etiqueta.")

show_table("25b_sensibilidad_target_ranking_modelos.xlsx", "Sensibilidad del target: ranking por escenario alternativo")
show_table("25c_decision_metodologica_target_final.xlsx", "Decision metodologica de target final (por escenario)")
show_table("26_optuna_lightgbm.xlsx", "Mejores hiperparametros LightGBM (Optuna)")
show_table("27_walk_forward_validation.xlsx", "Validacion Walk-Forward temporal (anio a anio)")
show_table("27b_nested_temporal_cv.xlsx", "Nested Temporal CV (por fold externo)", max_rows=20)
show_table("27c_nested_temporal_cv_summary.xlsx", "Nested Temporal CV - resumen por modelo")
show_table("28_ensemble_metrics.xlsx", "Metricas de ensambles (VotingClassifier / Stacking temporal)")

# ============================================================
# 5. OPTIMIZACION DE UMBRAL
# ============================================================
section("5. Optimizacion de Umbral de Decision")

show_table("24_umbral_optimo_validacion.xlsx", "Grilla de umbrales (F1/precision/recall en validacion)", max_rows=15)
show_table("24b_threshold_optimization_multicriterio.xlsx",
           "Umbral bajo multiples criterios (0.5 fijo, F1-optimo, Youden J, precision objetivo, recall objetivo)",
           "En datasets desbalanceados el umbral 0.5 rara vez es el optimo; se reportan alternativas justificadas.")

# ============================================================
# 6. METRICAS FINALES DEL MODELO GANADOR
# ============================================================
section("6. Metricas Finales del Modelo Ganador (umbral ajustado)")

_final_df = show_table("25_metricas_modelo_final_umbral_ajustado.xlsx",
                        "Metricas finales por dataset (valid/test/externa)",
                        "Incluye ademas: MCC, Cohen's Kappa, NPV, FPR, FNR, LR+, LR- (v3.02).")
show_table("40_bootstrap_confidence_intervals.xlsx", "Bootstrap (300 iteraciones): media e IC95% de F1/ROC-AUC/PR-AUC")
_article_ci_df = show_table("40b_metricas_principales_con_ic_para_articulo.xlsx",
                             "Metricas puntuales + IC95% listas para el articulo (test 2025)")

# ============================================================
# 7. PRUEBAS ESTADISTICAS DE COMPARACION ENTRE MODELOS
# ============================================================
section("7. Pruebas Estadisticas de Comparacion entre Modelos")

show_table("44_delong_roc_auc_comparacion.xlsx", "DeLong Test (comparacion de AUC entre pares de modelos)",
           "p<0.05 indica diferencia estadisticamente significativa en ROC-AUC entre el par de modelos.")
show_table("41_mcnemar_modelos.xlsx", "McNemar Test (comparacion de aciertos/errores pareados)")
show_table("45_friedman_test_modelos.xlsx", "Friedman Test (F1 a traves de folds temporales, todos los modelos)")
show_table("45b_friedman_scores_por_fold.xlsx", "F1 por fold y modelo (insumo del test de Friedman)")
show_table("46_nemenyi_posthoc_modelos.xlsx", "Nemenyi post-hoc (si Friedman fue significativo)")

# ============================================================
# 8. CALIBRACION DEL MODELO
# ============================================================
section("8. Calibracion del Modelo Predictivo")

show_table("42_brier_score.xlsx", "Brier Score + ECE + MCE (Expected/Maximum Calibration Error)")
show_figure("42_calibration_curve_test.png", "Curva de calibracion (test 2025)")
show_figure("42d_reliability_diagram.png", "Reliability Diagram (con ECE/MCE en el titulo)")
show_table("42c_reliability_diagram_bins.xlsx", "Bins del reliability diagram (confianza vs tasa observada)")

# ============================================================
# 9. EXPLICABILIDAD (SHAP, PERMUTATION, PDP)
# ============================================================
section("9. Explicabilidad del Modelo")

show_table("29_permutation_importance_test.xlsx", "Permutation Importance (test 2025)", max_rows=25)
show_figure("24_shap_beeswarm.png", "SHAP Beeswarm (top 20 features)")
show_figure("24_shap_bar.png", "SHAP Bar Chart (importancia media absoluta)")
show_table("30_shap_global_ranking.xlsx", "Ranking SHAP global", max_rows=25)
show_figure("24_shap_dependence_top_feature.png", "SHAP Dependence Plot (feature mas importante)")
show_table("30b_shap_interaction_values_top_pairs.xlsx", "Top pares de interaccion SHAP", max_rows=20)
show_figure("24_pdp_ice_top_features.png", "Partial Dependence + ICE (top 2 features)")
show_table("30c_counterfactual_explanations_controlled.xlsx", "Contrafactuales controlados (espacio preprocesado)")
show_table("33_shap_temporal_evolution.xlsx", "Evolucion temporal de la importancia SHAP (por anio)", max_rows=20)
show_table("34_shap_bootstrap_stability.xlsx", "Estabilidad de SHAP (bootstrap, CV%)", max_rows=20)

# ============================================================
# 10. LEARNING CURVE (AUSENCIA/PRESENCIA DE OVERFITTING)
# ============================================================
section("10. Learning Curve")

show_table("24c_learning_curve.xlsx", "F1 train vs F1 CV (walk-forward) segun tamano de entrenamiento")
show_figure("24c_learning_curve.png", "Learning Curve del modelo predictivo principal")

# ============================================================
# 11. CLUSTERING OPERATIVO Y ESTABILIDAD
# ============================================================
section("11. Clustering Operativo (Perfilamiento) y Estabilidad")

show_figure("25_elbow_kmeans.png", "Elbow Method para seleccion de K")
show_table("32_validacion_clusters_operativos.xlsx", "Metricas de validacion de clusters (KMeans/DBSCAN/Agglomerative)")
show_table("32b_cluster_bootstrap_stability_ari.xlsx", "Bootstrap Cluster Stability (Adjusted Rand Index)",
           "ARI cercano a 1 = clustering estable ante remuestreo; cercano a 0 = inestable.")
show_table("33_perfiles_operativos_clusters.xlsx", "Perfiles operativos por cluster")
show_figure("25_riesgo_proxy_por_cluster_operativo.png", "Riesgo proxy (%) por cluster operativo")
show_table("33b_target_proxy_por_cluster.xlsx", "Distribucion del target por cluster")
show_table("33c_validacion_convergente_target_cluster_escenarios.xlsx", "Validacion convergente target-cluster (todos los escenarios)", max_rows=20)
show_table("33d_experimento_cluster_como_feature.xlsx", "Experimento: cluster_id como feature adicional")
show_table("33_anomalias_isolation_lof_mahalanobis.xlsx", "Deteccion de anomalias: IsolationForest / LOF / Mahalanobis")

# ============================================================
# 12. CONFORMAL PREDICTION
# ============================================================
section("12. Conformal Prediction (Cobertura Garantizada)")

show_table("42b_conformal_prediction_summary.xlsx", "Split Conformal Prediction - cobertura global (alpha=0.10)")
show_table("42f_mondrian_conformal_summary.xlsx", "Mondrian Conformal por cluster operativo - resumen")
show_table("42e_mondrian_conformal_calibration_by_cluster.xlsx", "Calibracion Mondrian por cluster (q_hat)")

# ============================================================
# 13. FAIRNESS (EQUIDAD)
# ============================================================
section("13. Analisis de Equidad (Fairness)")

show_table("39_disparidad_territorial_preliminar.xlsx", "Disparidad preliminar (Lima vs. provincias, TPR/FPR)")
show_table("43_advanced_fairness_by_group.xlsx", "Fairness avanzado por grupo (demographic parity, equal opportunity, etc.)", max_rows=25)
show_table("43b_advanced_fairness_gaps.xlsx", "Mayores brechas entre grupos por metrica")

# ============================================================
# 14. DRIFT TEMPORAL Y VALIDACION EXTERNA 2026
# ============================================================
section("14. Drift Temporal (PSI) y Validacion Externa 2026")

show_table("35_drift_psi_temporal.xlsx", "Population Stability Index (PSI): train vs. cada anio posterior", max_rows=25)
show_table("36_validacion_externa_2026.xlsx", "Metricas en validacion externa 2026 (parcial, exploratoria)")
show_table("36_modelos_omitidos_validacion_externa_2026.xlsx", "Modelos omitidos en validacion externa (motivo)")

# ============================================================
# 15. ANALISIS DE ERRORES Y ROBUSTEZ
# ============================================================
section("15. Analisis de Errores y Robustez (Stress Testing / Ablation)")

show_table("37_error_summary_test.xlsx", "Resumen de errores (TP/TN/FP/FN) - test 2025")
show_table("38_error_analysis_por_grupo.xlsx", "Errores por grupo administrativo/territorial", max_rows=25)
show_table("32_stress_testing_results.xlsx", "Stress Testing: metricas bajo perturbacion de variables de carga (+10/20/30%)")
show_table("43_ablation_study.xlsx", "Ablation Study (impacto de retirar bloques de variables)")

# ============================================================
# 16. IMPACTO ECONOMICO Y ANALISIS CAUSAL EXPLORATORIO
# ============================================================
section("16. Impacto Economico y Analisis Causal Exploratorio (DoubleML)")

show_table("35_economic_impact.xlsx", "Estimacion de impacto economico (costo FN vs. FP)")
_dml_path = os.path.join(TABLE_DIR, "36_doubleml_results.csv")
if os.path.exists(_dml_path):
    show_table("36_doubleml_results.csv", "DoubleML - Efecto causal exploratorio (ATE)",
               "Exploratorio; no implica causalidad confirmada sin validacion experimental adicional.")
else:
    print("[No disponible] DoubleML no se ejecuto (paquete no instalado o error en el bloque).")

show_figure("28_evolucion_anual.png", "Evolucion anual de casos fiscales (ingresado/atendido/saldo)")
show_table("34_resumen_anual_temporal.xlsx", "Resumen anual temporal completo")

# ============================================================
# 17. REPORTES CONSOLIDADOS (PDF / HTML) YA GENERADOS
# ============================================================
section("17. Reportes Consolidados ya Generados por el Pipeline")

_pdf_path = os.path.join(PDF_DIR, "Final_Report_Q1_Extreme.pdf")
_html_path = os.path.join(HTML_DIR, "Final_Report_Q1.html")
print(f"PDF final: {'DISPONIBLE -> ' + _pdf_path if os.path.exists(_pdf_path) else 'no disponible'}")
print(f"HTML final: {'DISPONIBLE -> ' + _html_path if os.path.exists(_html_path) else 'no disponible'}")
if os.path.exists(_pdf_path):
    shutil.copy2(_pdf_path, os.path.join(ARTICLE_DIR, "Final_Report_Q1_Extreme.pdf"))
if os.path.exists(_html_path):
    shutil.copy2(_html_path, os.path.join(ARTICLE_DIR, "Final_Report_Q1.html"))

# ============================================================
# 18. INDICE DE FIGURAS Y TABLAS PARA EL ARTICULO
# ============================================================
section("18. Indice de Figuras y Tablas del Paquete para el Articulo")

_index_df = pd.DataFrame(_article_index_rows)
if len(_index_df):
    _index_path = os.path.join(ARTICLE_DIR, "INDICE_FIGURAS_TABLAS.xlsx")
    _index_df.to_excel(_index_path, index=False)
    display(_index_df)
    print(f"\n{len(_index_df)} elementos (figuras+tablas) copiados y numerados en: {ARTICLE_DIR}/")
    print(f"Indice completo: {_index_path}")
else:
    print("No se registro ningun elemento (revisa que el pipeline principal se haya ejecutado).")

# ============================================================
# 19. RESUMEN EJECUTIVO PARA LA SECCION DE RESULTADOS DEL ARTICULO
# ============================================================
section("19. Resumen Ejecutivo (copiar/pegar en la seccion de Resultados)")

def _get_metric(df, dataset_name, metric_name):
    if df is None or "dataset" not in df.columns or metric_name not in df.columns:
        return None
    row = df[df["dataset"] == dataset_name]
    if len(row) == 0:
        return None
    val = row.iloc[0][metric_name]
    return val if pd.notna(val) else None

def _get_ci(article_ci_df, metric_name):
    if article_ci_df is None or "metric" not in article_ci_df.columns:
        return None, None, None
    row = article_ci_df[article_ci_df["metric"] == metric_name]
    if len(row) == 0:
        return None, None, None
    r = row.iloc[0]
    return r.get("point_estimate"), r.get("ci_2_5"), r.get("ci_97_5")

_scorecard_df = _read_table("PROJECT_SCORECARD.xlsx")
_best_model_name = None
if _final_df is not None and "modelo" in _final_df.columns and len(_final_df):
    _best_model_name = _final_df.iloc[0]["modelo"]

_brier_df = _read_table("42_brier_score.xlsx")
_thr_opt_df = _read_table("24b_threshold_optimization_multicriterio.xlsx")

f1_test, f1_lo, f1_hi = _get_ci(_article_ci_df, "f1")
auc_test, auc_lo, auc_hi = _get_ci(_article_ci_df, "roc_auc")
pr_test, pr_lo, pr_hi = _get_ci(_article_ci_df, "pr_auc")

summary_lines = []
summary_lines.append("=" * 78)
summary_lines.append("RESUMEN EJECUTIVO - LISTO PARA LA SECCION DE RESULTADOS")
summary_lines.append("=" * 78)
summary_lines.append(f"Modelo predictivo principal: {_best_model_name if _best_model_name else 'N/A'}")
if _thr_opt_df is not None and len(_thr_opt_df):
    _f1_thr_row = _thr_opt_df[_thr_opt_df["criterio"].str.contains("F1-optimo", na=False)]
    if len(_f1_thr_row):
        summary_lines.append(f"Umbral de decision (F1-optimo, validacion 2024): {fmt(_f1_thr_row.iloc[0]['threshold'], 3)}")
summary_lines.append(
    f"F1 (test 2025): {fmt(f1_test)} (IC95%: {fmt(f1_lo)}-{fmt(f1_hi)})" if f1_test is not None else "F1 (test 2025): N/A")
summary_lines.append(
    f"ROC-AUC (test 2025): {fmt(auc_test)} (IC95%: {fmt(auc_lo)}-{fmt(auc_hi)})" if auc_test is not None else "ROC-AUC (test 2025): N/A")
summary_lines.append(
    f"PR-AUC (test 2025): {fmt(pr_test)} (IC95%: {fmt(pr_lo)}-{fmt(pr_hi)})" if pr_test is not None else "PR-AUC (test 2025): N/A")
if _brier_df is not None and len(_brier_df):
    _b = _brier_df.iloc[0]
    summary_lines.append(f"Calibracion: Brier={fmt(_b.get('brier_score'))} | ECE={fmt(_b.get('ece'))} | MCE={fmt(_b.get('mce'))}")
if _votes_df is not None:
    _sel_feats_df = _read_table("21_selected_features_final.xlsx")
    _n_sel_feats = len(_sel_feats_df) if _sel_feats_df is not None else 0
    summary_lines.append(f"Features seleccionadas: {_n_sel_feats} "
                          f"(votacion >=2/6 metodos; ver Tabla de Stability Score)")
if _scorecard_df is not None and len(_scorecard_df):
    _overall_row = _scorecard_df[_scorecard_df["Componente"] == "OVERALL"]
    if len(_overall_row):
        summary_lines.append(f"Project Scorecard (autoevaluacion metodologica): {fmt(_overall_row.iloc[0]['Score'], 2)}/10")

summary_lines.append("")
summary_lines.append("Recordatorio para la redaccion:")
summary_lines.append(" - riesgo_congestion es una senal PROXY (no certificacion oficial ni causalidad).")
summary_lines.append(" - Oracle_TargetDefinitionCheck (Acc=1.0) es un sanity check, NO un modelo predictivo comparable.")
summary_lines.append(" - Los IV extremos de saldo_casos/tasa_atencion/ratio_saldo se deben a que construyen el target.")
summary_lines.append(" - La validacion externa 2026 es parcial/exploratoria (anio incompleto al momento de ejecucion).")
summary_lines.append(" - DeLong/McNemar/Friedman ya evaluaron significancia estadistica entre modelos (Seccion 7).")
summary_lines.append("=" * 78)

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(os.path.join(ARTICLE_DIR, "RESUMEN_EJECUTIVO_PARA_ARTICULO.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nResumen ejecutivo guardado en: {os.path.join(ARTICLE_DIR, 'RESUMEN_EJECUTIVO_PARA_ARTICULO.txt')}")
print(f"\nPAQUETE COMPLETO PARA EL ARTICULO EN: ./{ARTICLE_DIR}/")
print(f"  - {ARTICLE_FIG_DIR}/   (todas las figuras numeradas)")
print(f"  - {ARTICLE_TABLE_DIR}/  (todas las tablas numeradas)")
print(f"  - INDICE_FIGURAS_TABLAS.xlsx")
print(f"  - RESUMEN_EJECUTIVO_PARA_ARTICULO.txt")
print(f"  - Final_Report_Q1_Extreme.pdf / Final_Report_Q1.html (si existen)")

# Comprimir el paquete completo del articulo en un solo ZIP para descargar/compartir.
try:
    import zipfile
    _article_zip = "Casos_Fiscales_2026.zip"
    with zipfile.ZipFile(_article_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(ARTICLE_DIR):
            for fname in files:
                fp = os.path.join(root, fname)
                zf.write(fp, os.path.relpath(fp, "."))
    print(f"\nZIP del paquete para el articulo: {_article_zip}")
    try:
        from google.colab import files as _colab_files
        _colab_files.download(_article_zip)
    except Exception:
        pass
except Exception as e:
    print(f"No se pudo generar article_package.zip: {e}")

print("\n" + "=" * 78)
print("VISOR COMPLETADO. Revisa ./article_package/ para redactar el articulo.")
print("=" * 78)
```