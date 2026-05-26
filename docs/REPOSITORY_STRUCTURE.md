# Repository Structure Guide

## Overview

This document explains the organization of the Bike Rental Demand Forecasting project. The repository is organized into logical functional areas to help contributors quickly understand the codebase.

---

## Directory Structure

### 📊 `data/` - Datasets

Contains all raw and processed data files.

- **`raw/`** - Original source data
  - `daily_rent_detail.csv` - Daily rental transaction data
  - `station_list.csv` - Station metadata
  - `usage_frequency.csv` - Station usage statistics
  - `weather.csv` - Weather data (temperature, humidity, etc.)

- **`processed/`** - Cleaned and transformed datasets
  - `encoded_train.csv` - One-hot encoded training set (70% of data)
  - `encoded_validation.csv` - One-hot encoded validation set (15% of data)
  - `encoded_test.csv` - One-hot encoded test set (15% of data)
  - `encoded_feature_names.csv` - List of feature names after encoding
  - `daily_rentals_top20*.csv` - Intermediate processed files
  - `analysis/` - Aggregated metrics and analysis results
  - `with_lag/` - Features including lag variables (lag_1, lag_7)
  - `without_lag/` - Features without lag variables (for comparison)

**See also:** `docs/ENCODED_DATASETS.md` for encoding details

---

### 💻 `src/` - Source Code & Utilities

Application code and helper functions.

- **`data_processing/`** - Data pipeline scripts
  - `02_build_top20_daily_dataset.py`
    - Filters top 20 stations
    - Creates daily aggregated rental data
    - Input: `data/raw/*.csv`
    - Output: `data/processed/daily_rentals_top20.csv`

  - `03_analyze_feature_correlations.py`
    - Analyzes feature correlations
    - Generates correlation heatmaps
    - Helps identify useful features

  - `04_build_reduced_feature_dataset.py`
    - Reduces feature set
    - Prepares numerical features
    - Output: `data/processed/daily_rentals_top20_reduced.csv`

  - `05_create_encoded_dataset.py`
    - One-hot encodes station IDs
    - Adds lag features (optional)
    - Performs train/val/test split
    - Output: `data/processed/encoded_*.csv`

- **`data_statistics.py`** - Data exploration utilities
  - Summary statistics functions
  - Used for data profiling

**Usage:**
```bash
python src/data_processing/02_build_top20_daily_dataset.py
python src/data_processing/05_create_encoded_dataset.py
```

---

### 🤖 `modelling/` - Machine Learning Models

All model implementations and training pipelines.

#### Structure

```
modelling/
├── common/                    # Shared utilities
│   ├── config.py             # Configuration constants (paths, parameters)
│   ├── preprocessing.py       # Data loading and preprocessing
│   ├── metrics.py            # Evaluation metrics
│   └── ...
│
├── 00_dummy_regressor/        # Baseline: predicts mean
│   ├── train_dummy.py        # Training script
│   ├── model/                # Saved model artifacts
│   └── results/              # Results and metrics
│
├── 01_linear_regression/      # Linear models
├── 02_ridge_regression/       # Ridge (L2 regularization)
├── 03_lasso_regression/       # Lasso (L1 regularization)
├── 04_decision_tree/          # Decision tree regressor
├── 05_knn_regressor/          # K-Nearest Neighbors
├── 06_random_forest/          # 🥇 Best performing model
├── 07_gradient_boosting/      # 🥈 Second best model
├── 08_neural_network/         # Deep learning approach
│
└── 99_model_comparison/       # Comparison & ranking
    ├── model_comparison.py    # Generates performance tables
    └── results/              # Comparison outputs
```

#### Key Files in Each Model Directory

- **`train_<model_name>.py`** - Training script
  - Loads encoded datasets
  - Trains model with both lag/without_lag variants
  - Saves model and predictions
  - Example: `python -m modelling.00_dummy_regressor.train_dummy --experiment all`

- **`model/<variant>/`** - Saved model artifacts
  - `without_lag/model.pkl` - Trained model
  - `with_lag/model.pkl` - Trained model with lag features

- **`results/<variant>/`** - Training results
  - `test_predictions.csv` - Predictions on test set
  - `metrics.json` - Performance metrics (RMSE, R², MAE)
  - `train_history.json` - Training history (for NN)

#### Experiments

Each model supports two experimental variants:
- **`without_lag`** - Features without lag variables
- **`with_lag`** - Features including lag_1 and lag_7

Run with: `--experiment all` (both) or `--experiment without_lag`/`--experiment with_lag` (specific)

---

### 📈 `analysis/` - Analysis & Visualization

Scripts for generating insights, analysis reports, and presentation-ready plots.

#### Analysis Scripts

- **`generate_time_error_insights.py`**
  - Analyzes prediction errors across time periods
  - Output: Temporal error breakdown and visualizations

- **`generate_weather_error_insights.py`**
  - Analyzes how weather conditions affect prediction errors
  - Bins data by temperature, humidity, weather type
  - Output: Weather-specific error analysis

- **`generate_additional_question_insight_plots.py`**
  - Domain-specific analysis questions
  - Custom insights and plots

- **`generate_question_backup_plots.py`**
  - Backup analysis scripts
  - Alternative visualizations

#### Plot Scripts

- **`plot_top3_rmse_by_split.py`**
  - Compares top 3 models across train/val/test splits
  - Shows RMSE for with_lag vs without_lag

- **`plot_top3_scatter.py`**
  - Scatter plots of predictions vs actual values
  - Shows model accuracy for top 3 models

- **`plot_lag_vs_without_lag_rmse.py`**
  - Compares performance with/without lag features
  - Per-model comparison

- **`plot_without_lag_rmse_comparison.py`**
  - Detailed comparison of non-lag models

- **`plot_related_work_comparison.py`**
  - Compares results with related literature
  - Benchmarking analysis

#### Output Directories

- **`plots/`** - Generated visualization images (PNG/PDF)
- **`insights/`** - Detailed analysis reports (HTML/text files)
- **`results/`** - Analysis metrics and data (CSV/JSON)

**Usage:**
```bash
cd analysis/
python generate_time_error_insights.py
python plot_top3_rmse_by_split.py
```

---

### 📄 `paper/` - Academic Paper

LaTeX source for the research paper.

- **`paper.tex`** - Main document
- **`*.tex`** - Individual sections (introduction, methodology, results, etc.)
- **`model_comparison.py`** - Script to generate comparison tables
- **`Literature/`** - Bibliography and literature files
- **`notes/`** - Research notes and draft sections

---

### 🖼️ `presentation_figures/` - Presentation Graphics

Polished figures for presentations and publications.

- **`additional_question_insights/`** - Plots for research questions
- **`question_backup_plots/`** - Backup visualizations
- **`time_question_insights/`** - Temporal analysis figures
- **`time_question_insights_detailed/`** - Detailed temporal breakdowns
- **`weather_question_insights/`** - Weather-specific figures
- **`related_work_comparison.csv`** - Related work data

---

### 📚 `docs/` - Documentation

Reference documentation and guides.

- **`README.md`** - Quick start and overview
- **`REPOSITORY_STRUCTURE.md`** (this file) - Detailed structure guide
- **`ENCODED_DATASETS.md`** - Data encoding pipeline explanation
- **`PROJECT_ANALYSIS.md`** - Comprehensive analysis report with results

---

### 🔧 Root-Level Files

- **`run_pipeline.py`** - Main automation script
  - Orchestrates entire pipeline
  - Data processing → Model training → Comparison
  - Flags at top control which steps to run
  - Usage: `python run_pipeline.py`

- **`requirements.txt`** - Python package dependencies
  - Install with: `pip install -r requirements.txt`

---

## How to Navigate the Project

### I'm new - where do I start?
1. Read `docs/README.md` for overview
2. Read this file (`docs/REPOSITORY_STRUCTURE.md`)
3. Check `run_pipeline.py` to understand the workflow

### I want to understand the data
1. `src/data_processing/` - Data transformation pipeline
2. `docs/ENCODED_DATASETS.md` - Encoding details
3. `data/processed/` - Actual datasets

### I want to train/modify models
1. `modelling/` - Choose a model directory
2. Look at `modelling/common/` - Understand shared utilities
3. Edit the `train_<model_name>.py` script
4. Run: `python -m modelling.XX_modelname.train_<model_name>`

### I want to see results
1. `docs/PROJECT_ANALYSIS.md` - Detailed performance metrics
2. `analysis/` - Run analysis scripts for visualizations
3. `modelling/XX_modelname/results/` - Raw results files

### I want to add new visualizations
1. Create a new script in `analysis/` following naming convention
2. Output images to `analysis/plots/`
3. Output reports to `analysis/insights/`
4. Output data to `analysis/results/`

### I want to understand a specific feature
1. Check if it's in `modelling/common/config.py`
2. Look for related processing in `src/data_processing/`
3. See usage in relevant model training script

---

## File Naming Conventions

- **Data processing scripts:** `XX_description.py` (numbered sequentially)
- **Model training scripts:** `train_<model_name>.py`
- **Analysis/plot scripts:** `generate_<type>.py` or `plot_<type>.py`
- **Data files:** `<dataset_name>_<variant>.csv` (e.g., `encoded_test.csv`)
- **Results files:** `test_predictions.csv`, `metrics.json`

---

## Data Flow

```
Raw Data (data/raw/)
    ↓
[02_build_top20_daily_dataset.py] → daily_rentals_top20.csv
    ↓
[03_analyze_feature_correlations.py] → correlation analysis
    ↓
[04_build_reduced_feature_dataset.py] → daily_rentals_top20_reduced.csv
    ↓
[05_create_encoded_dataset.py] → encoded_train/val/test.csv
    ↓
Model Training (modelling/XX_*/train_*.py)
    ↓
Results (modelling/XX_*/results/)
    ↓
Analysis & Visualization (analysis/*.py)
    ↓
Presentation Figures & Reports
```

---

## Key Results Summary

**Best Model:** Random Forest
- Test RMSE: 30.31
- Test R²: 0.406
- MAE: 22.87

**Second Best:** Gradient Boosting
- Test RMSE: 30.57
- Test R²: 0.396
- Validation R²: 0.474 (highest)

See `docs/PROJECT_ANALYSIS.md` for complete comparison table.

---

## Important Notes

1. **Chronological Split:** Data is split chronologically (70% train / 15% val / 15% test), not randomly
2. **Lag Features:** Models are evaluated with and without lag features (lag_1 and lag_7)
3. **Encoding:** Station IDs are one-hot encoded; feature list is in `encoded_feature_names.csv`
4. **Top 20:** All analysis uses only the top 20 stations by rental volume
5. **Artifacts:** Model files and results are version-controlled in respective `model/` and `results/` directories

---

For detailed technical information, see individual section documentation:
- Data processing: `docs/ENCODED_DATASETS.md`
- Project analysis: `docs/PROJECT_ANALYSIS.md`
- Main README: `docs/README.md`
