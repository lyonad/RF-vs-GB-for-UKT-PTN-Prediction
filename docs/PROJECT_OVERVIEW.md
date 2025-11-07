# Project Overview: RF vs GB for UKT-PTN Prediction

This document gives you a concise, code-level understanding of the repository, covering architecture, data flow, training and evaluation pipeline, the Flask web app, and how everything fits together.

## What this project does

Predicts Indonesian public university tuition fees (UKT) across 11 tiers using tree-based models (RandomForest, XGBoost, CatBoost, LightGBM), with uncertainty via quantile regression and conformal prediction. It includes:
- End-to-end research pipeline (data → models → evaluation → reports/plots)
- Saved trained models and analysis artifacts
- A Flask web app for interactive predictions and a JSON API

## Tech stack

- Python: pandas, numpy, scikit-learn, xgboost, catboost, lightgbm, matplotlib, seaborn, plotly
- Web: Flask, Jinja2, Chart.js, HTML/CSS/JS
- Tests: unittest

## Repository map (purpose)

- `src/`
  - `data_preprocessor.py`: Load/create data, encode categoricals, scale features, train/test split
  - `models.py`: Multi-output regressors for RF/XGB/CatBoost/LGBM; quantile models per algorithm
  - `conformal_prediction.py`: Split conformal prediction with shared calibration across outputs
  - `evaluation.py`: Metrics (RMSE, MAE, R², MAPE, Pearson), coverage, plots (PNG + interactive HTML), reporting
  - `main.py`: Orchestrates the full pipeline and writes outputs/models
  - `config.py`: Defaults (paths, seed, hyperparams, quantiles)
- `utils/visualization.py`: Plots for distributions, importances, intervals, quantile lines; accuracy analysis helper
- `webapp/`: Flask app with UI, templates, static assets, and JSON endpoints
- `tests/`: Unit tests for preprocessing, models, conformal, evaluator
- `Data/`: `data.csv` input (auto-synthesized if absent by DataPreprocessor)
- `results/`: CSV/HTML/plots, arrays: comparison/report/visualizations/accuracy
- `models/`: Saved trained models (one file per algorithm)

## End-to-end pipeline (src/main.py)

1. Data
   - Load `Data/data.csv`; if missing, a realistic sample dataset is synthesized.
   - Infer target columns as those starting with `UKT-`.
   - Encode categoricals with LabelEncoder, fill missing, split (train/test), and StandardScaler on features.

2. Train models (`TuitionFeePredictor`)
   - MultiOutputRegressor wrapped estimators: RandomForest, XGBoost, CatBoost, LightGBM
   - Train standard point models and per-quantile models (0.1/0.5/0.9) using native quantile loss when available

3. Conformal prediction
   - Build `MultiOutputConformalPredictor` for each trained model (split conformal; shared calibration split to ensure fairness across outputs)
   - For each model: fit, then produce (y_pred, lower, upper)

4. Evaluation & visualization (`ModelEvaluator` + utils)
   - Compute metrics per model; compare across models
   - Coverage (percent y_true within [lower, upper])
   - Plots: comparisons, importances, intervals (per sample model), quantile lines
   - Accuracy analysis for best model (within-threshold %, MAPE/MDAPE, errors)

5. Persist artifacts
   - `results/model_comparison.csv`, `results/research_report.md`
   - Per-model pickles, e.g. `models/catboost_model.pkl`
   - Conformal predictors saved via `results/conformal_predictors.pkl`

## Key modules at a glance

- DataPreprocessor
  - Inputs: `Data/data.csv` (or synthesizes realistic sample)
  - Outputs: X (encoded & scaled later), y (multiple UKT targets)
- TuitionFeePredictor
  - Trains 4 algorithms (RF/XGB/CatBoost/LGBM) as multi-output regressors
  - Builds quantile models: {0.1, 0.5, 0.9} per algorithm
  - Saves individual model files
- Conformal predictors
  - Split conformal with shared calibration; per-output predictors under the hood
  - Predict returns `(y_pred, lower_bounds, upper_bounds)`
- ModelEvaluator
  - Metrics (per-output + overall), coverage, plots, interactive comparison, report generator

## Web app (webapp/)

- `app.py` (Flask)
  - Loads: `models/catboost_model.pkl` (point model)
  - Prepares encoders/scaler from `Data/data.csv` using DataPreprocessor
  - Endpoints:
    - `GET /` UI form (dropdowns sourced from data)
    - `POST /api/predict` → returns all 11 UKT tier predictions, optionally with intervals
    - `GET /api/model-info` → model metadata + metrics snapshot
    - `GET /api/health` → status checks
    - `GET /about`, `GET /documentation` → static pages
  - Frontend (`templates/`, `static/js/main.js`, `static/css/style.css`) uses Chart.js for visualizing predictions and intervals.

### Important note about conformal intervals in the web app

- Training saves conformal predictors to `results/conformal_predictors.pkl` (a dict keyed by model names like `"CatBoost"`).
- Current web app looks for `models/catboost_conformal.pkl` and calls `predict_interval()`, which differs from the library’s API that returns `(y_pred, lower, upper)` via `predict()`.
- Suggested options:
  1) Update web app to load `results/conformal_predictors.pkl`, select the `"CatBoost"` entry, and call `predict()` to get intervals.
  2) Or export a `models/catboost_conformal.pkl` containing just the CatBoost conformal predictor and adjust method calls.

This mismatch only affects intervals; point predictions work with `models/catboost_model.pkl`.

## How to run

- One-shot research run (Windows):
  - `setup.bat` (creates `.venv`, installs `requirements.txt`)
  - `run_research.bat` (activates venv, runs `python src/main.py`)

- Manual
  - Create venv; install `requirements.txt`
  - `python src/main.py`

- Web app
  - Ensure models exist (e.g., `models/catboost_model.pkl`); if not, run the research pipeline first
  - From project root: `python webapp/app.py` (or use `.venv\Scripts\python.exe webapp\app.py`)
  - Open http://localhost:5000

## Tests

- Run: `python -m pytest tests/test_project.py -v` (or use unittest directly)
- Coverage: Preprocessing, model fit/predict, quantile outputs, conformal intervals, metrics/compare

## Outputs (examples)

- Results: `results/model_comparison.csv`, `results/research_report.md`, `results/visualizations/*`, `results/accuracy_analysis.npy`
- Models: `models/{randomforest|xgboost|catboost|lightgbm}_model.pkl`
- Conformal: `results/conformal_predictors.pkl`

## Known gaps & quick fixes

- Web app interval loading/API mismatch (see note above).
- `webapp/requirements.txt` only lists Flask bits; the app imports numpy/pandas/scikit-learn/joblib. In practice it relies on the root environment. If running `webapp` standalone, add these to `webapp/requirements.txt`.
- Error template references `models/catboost_best.pkl`; current pipeline saves `models/catboost_model.pkl` instead.

## Extend and maintain

- Add hyperparameter tuning (Optuna) per model
- Log experiment runs (e.g., MLflow) and persist metrics
- Package the training pipeline as a CLI entry point
- Containerize web app with model artifacts mounted/packaged
- Add CI (pytest) and pre-commit hooks (black/isort/ruff)
