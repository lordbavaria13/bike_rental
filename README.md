# Bike Rental Demand Forecasting Project

## Project Goal

The project predicts the daily number of bike rentals per station using machine learning.

The final target variable is `total_rentals`.

**Workflow**: Raw bike rental and weather data → Data cleaning & feature engineering → One-hot encoded features with lag variables → Training & comparison of multiple regression models → Analysis & visualization of results.

---

## 📍 Navigation

**First time here?** Start with:
1. **[INDEX.md](INDEX.md)** - Quick navigation guide for this repository
2. **[docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)** - Detailed structure explanation
3. Then come back here for quick start

**Just want results?** → Check [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md)

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The project requires the following packages:

| Package | Version |
|---|---|
| matplotlib | 3.10.8 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| seaborn | 0.13.2 |
| joblib | 1.5.3 |
| pyarrow | 24.0.0 |

Python 3.10 or higher is recommended.

---

## Project Structure

```
bike_rental/
├── data/                                 # Raw and processed datasets
│   ├── raw/                              # Original data files
│   │   ├── daily_rent_detail.csv
│   │   ├── station_list.csv
│   │   ├── usage_frequency.csv
│   │   └── weather.csv
│   └── processed/                        # Cleaned and engineered datasets
│       ├── encoded_train.csv             # One-hot encoded training data
│       ├── encoded_validation.csv        # One-hot encoded validation data
│       ├── encoded_test.csv              # One-hot encoded test data
│       ├── daily_rentals_top20*.csv      # Intermediate processing stages
│       └── analysis/                     # Analysis results and metrics
│
├── src/                                  # Source code and utilities
│   ├── data_processing/                  # Data pipeline scripts
│   │   ├── 02_build_top20_daily_dataset.py
│   │   ├── 03_analyze_feature_correlations.py
│   │   ├── 04_build_reduced_feature_dataset.py
│   │   └── 05_create_encoded_dataset.py
│   └── data_statistics.py                # Data exploration utilities
│
├── modelling/                            # Machine Learning models
│   ├── common/                           # Shared utilities & config
│   ├── 00_dummy_regressor/               # Baseline model
│   ├── 01_linear_regression/
│   ├── 02_ridge_regression/
│   ├── 03_lasso_regression/
│   ├── 04_decision_tree/
│   ├── 05_knn_regressor/
│   ├── 06_random_forest/                 # Best performing model
│   ├── 07_gradient_boosting/             # Second best model
│   ├── 08_neural_network/
│   └── 99_model_comparison/              # Model comparison & ranking
│
├── analysis/                             # Analysis, insights & visualization
│   ├── generate_*.py                     # Scripts for generating insights
│   │   ├── generate_time_error_insights.py
│   │   ├── generate_weather_error_insights.py
│   │   ├── generate_additional_question_insight_plots.py
│   │   └── generate_question_backup_plots.py
│   ├── plot_*.py                         # Visualization scripts
│   │   ├── plot_top3_rmse_by_split.py
│   │   ├── plot_top3_scatter.py
│   │   ├── plot_lag_vs_without_lag_rmse.py
│   │   ├── plot_without_lag_rmse_comparison.py
│   │   └── plot_related_work_comparison.py
│   ├── plots/                            # Generated plot images
│   ├── insights/                         # Generated insight reports
│   └── results/                          # Analysis results and metrics
│
├── paper/                                # Academic paper & research notes
│   ├── paper.tex
│   ├── model_comparison.py
│   └── sections/                         # Individual LaTeX sections
│
├── presentation_figures/                 # Presentation-ready visualizations
│   ├── additional_question_insights/
│   ├── question_backup_plots/
│   ├── time_question_insights/
│   └── weather_question_insights/
│
├── docs/                                 # Project documentation
│   ├── ENCODED_DATASETS.md               # Encoding pipeline documentation
│   ├── PROJECT_ANALYSIS.md               # Comprehensive analysis report
│   └── README.md                         # (this file)
│
├── run_pipeline.py                       # Main pipeline automation script
├── requirements.txt                      # Python dependencies
└── README.md                             # Project overview
```

---

## Quick Start

### 1. Setup

```bash
# Activate virtual environment
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

### 2. Run the Full Pipeline

The easiest way to run everything is with `run_pipeline.py`:

```bash
python run_pipeline.py
```

Individual steps can be enabled/disabled by changing flags at the top of `run_pipeline.py`:

```python
STEP_1_DATA_PROCESSING = 1              # Enable/disable
RUN_00_DUMMY = 1
RUN_01_LINEAR = 1
# ... etc
RUN_07_GRADIENT_BOOSTING = 1
STEP_3_MODEL_COMPARISON = 1
```

### 3. Run Models Individually

```bash
# Data processing
python src/data_processing/02_build_top20_daily_dataset.py
python src/data_processing/05_create_encoded_dataset.py

# Model training
python -m modelling.00_dummy_regressor.train_dummy --experiment all
python -m modelling.01_linear_regression.train_linear_regression --experiment all
python -m modelling.06_random_forest.train_random_forest --experiment all
# ... etc for other models

# Model comparison
python -m modelling.99_model_comparison.model_comparison
```

---

## Analysis & Visualization

After training, generate insights and visualizations:

```bash
# Generate analysis reports and plots
cd analysis/

# Time-based error analysis
python generate_time_error_insights.py

# Weather-based error analysis
python generate_weather_error_insights.py

# Question-specific insights
python generate_additional_question_insight_plots.py
python generate_question_backup_plots.py

# Performance comparison plots
python plot_top3_rmse_by_split.py
python plot_top3_scatter.py
python plot_lag_vs_without_lag_rmse.py
python plot_without_lag_rmse_comparison.py
python plot_related_work_comparison.py
```

**Output locations:**
- `analysis/plots/` - Generated visualizations (PNG/PDF)
- `analysis/insights/` - Detailed analysis reports (HTML/TXT)
- `presentation_figures/` - Presentation-ready figures

---

## Key Results

**Best Model: Random Forest**
- Test RMSE: 30.31
- Test R²: 0.406
- Test MAE: 22.87

**Runner-up: Gradient Boosting**
- Test RMSE: 30.57
- Test R²: 0.396
- Validation R²: 0.474 (highest)

See `docs/PROJECT_ANALYSIS.md` for detailed performance tables.

---

## Documentation

- **`docs/PROJECT_ANALYSIS.md`** - Comprehensive analysis report with performance metrics for all 9 models
- **`docs/ENCODED_DATASETS.md`** - Explanation of data encoding pipeline and feature engineering
- **`modelling/common/`** - Shared utilities, config, and preprocessing functions
- **`paper/`** - Academic paper in LaTeX format with results, discussion, and references

---

## Navigation Guide for New Contributors

1. **Just want to run the code?** → Start with `run_pipeline.py`
2. **Want to understand the data?** → Check `src/data_processing/` and `docs/ENCODED_DATASETS.md`
3. **Want to train models?** → Go to `modelling/` and choose a model directory
4. **Want to see results & analysis?** → Go to `analysis/` and run the generate/plot scripts
5. **Want model performance details?** → Read `docs/PROJECT_ANALYSIS.md`
6. **Need configuration changes?** → Check `modelling/common/config.py`

---

## Notes

- All models use chronological time split (70% train / 15% validation / 15% test)
- Models are tested with both lag and non-lag feature variants
- Results are stored in `modelling/<model>/results/<variant>/`
- Models are stored in `modelling/<model>/model/<variant>/`
```

The `--experiment` flag accepts:
- `all` — runs both `without_lag` and `with_lag` variants
- `with_lag` — runs only the lag variant
- `without_lag` — runs only the no-lag variant

**Important:** Always run from the project root, never from inside
a subfolder. Scripts that import `modelling.common` require the
project root to be on the Python path.

---

## Data Processing Pipeline

### `02_build_top20_daily_dataset.py`

Builds the first processed dataset.

- Finds the top 20 start stations by total rentals
- Aggregates raw rental data to daily level
- Merges station and weather information

Main outputs:
- `data/processed/top_20_stations.csv`
- `data/processed/daily_rentals_top20.csv.gz`

---

### `03_analyze_feature_correlations.py`

Analyzes redundancy in the processed dataset.

- Computes a correlation matrix for all numeric predictors
- Creates a heatmap
- Detects and reports highly correlated variable pairs

Typical examples removed based on this analysis:
- `solarenergy` / `solarradiation`
- `feelslike` / `temp`
- `tempmin` / `feelslikemin`

Main output folder: `data/processed/analysis/`

---

### `04_build_reduced_feature_dataset.py`

Builds the final reduced modelling dataset.

- Removes leakage-prone count columns (member/casual, bike type)
- Keeps one target: `total_rentals`
- Keeps selected weather, time, and station variables
- Saves a station name mapping file

Final feature set:
`start_station_id`, `tempmax`, `humidity`, `precip`, `precipcover`,
`cloudcover`, `windspeed`, `visibility`, `sealevelpressure`,
`uvindex`, `sunset_minutes`, `month`, `weekday`, `year`,
`time_idx`, `snow`, `snowdepth`, `total_rentals`

Main outputs:
- `data/processed/daily_rentals_top20_reduced.csv`
- `data/processed/station_id_name_mapping.csv`

---

### `05_create_encoded_dataset.py`

Creates the final encoded datasets used for modelling.

- Applies one-hot encoding to `start_station_id` (20 binary columns)
- Creates two dataset variants:
  - **without_lag**: 35 features, no demand history
  - **with_lag**: 37 features, adds `total_rentals_lag_1` and
    `total_rentals_lag_7` (station-specific)
- Filters both variants to identical station-day keys for fair
  comparison
- Removes rows without full lag history

Main outputs:
- `data/processed/encoded/dataset_without_lag.csv`
- `data/processed/encoded/dataset_with_lag.csv`

---

## Modelling Common Utilities (`modelling/common/`)

All model scripts share the same utilities:

| File | Purpose |
|---|---|
| `config.py` | Paths, target column, split ratios, random seed |
| `preprocessing.py` | Data loading, feature selection, scaling |
| `split.py` | Chronological train/validation/test split (70/15/15) |
| `metrics.py` | MAE, RMSE, R², Median AE, MAPE, Explained Variance |
| `plotting.py` | Shared plot style for all models |
| `training.py` | Shared experiment runner used by all model scripts |
| `utils.py` | Directory creation, JSON and CSV saving |

---

## Model Folders

Each model folder follows the same structure:

```text
model_folder/
├── train_....py
├── results/
│   ├── without_lag/
│   │   ├── metrics.csv
│   │   ├── predictions.csv
│   │   └── plots/
│   └── with_lag/
│       ├── metrics.csv
│       ├── predictions.csv
│       └── plots/
└── model/
    ├── without_lag/
    └── with_lag/
```

Each model is evaluated on both the `without_lag` and `with_lag`
dataset variants. Results are saved separately per variant.

---

## Models

| Model | Scaling | Notes |
|---|---|---|
| `00_dummy_regressor` | no | Predicts training mean; minimum baseline |
| `01_linear_regression` | yes | Standard OLS regression |
| `02_ridge_regression` | yes | L2 regularization; tunes `alpha` |
| `03_lasso_regression` | yes | L1 regularization; tunes `alpha` |
| `04_decision_tree` | no | Tunes `max_depth`, `min_samples_leaf` |
| `05_knn_regressor` | yes | Tunes `n_neighbors`, `weights`, `p` |
| `06_random_forest` | no | Tunes `n_estimators`, `max_depth` |
| `07_gradient_boosting` | no | Tunes `n_estimators`, `learning_rate`, `max_depth` |
| `08_neural_network` | yes | MLPRegressor; tunes architecture and `alpha` |

Scaling is applied only where the model requires it. Tree-based
models (Decision Tree, Random Forest, Gradient Boosting) do not
need scaling because their splits are threshold-based.

---

## Evaluation Setup

All models are evaluated with the same strategy:

- **Split:** Chronological 70% train / 15% validation / 15% test
- **Tuning:** Best hyperparameters selected on validation set only
- **Final score:** Always reported on the held-out test set

Metrics reported for every model:

| Metric | Why used |
|---|---|
| MAE | Easy to interpret in rental units |
| RMSE | Main ranking metric; penalizes large errors |
| Median AE | Robust to extreme demand days |
| MAPE | Relative error; useful across stations with different demand levels |
| R² | Variance explained vs. mean baseline |
| Explained Variance | Checks whether predicted spread matches real spread |

---

## Model Comparison (`99_model_comparison/`)

After all models are trained, run the comparison script:

```bash
python -m modelling.99_model_comparison.model_comparison
```

This script collects all `metrics.csv` files, builds rankings, and
saves a full comparison including plots and summary tables.

Main outputs:
- `all_model_metrics.csv`
- `model_comparison_compact.csv`
- `model_ranking_validation_rmse.csv`
- `model_ranking_test_rmse.csv`
- `comparison_summary.json`

---

## Current Limitations

| Limitation | Details |
|---|---|
| Top-20 stations only | Results do not generalize to smaller stations |
| Daily aggregation | Hourly rush-hour peaks are not visible |
| Missing event data | Public events, strikes, road works not included |
| Temporal drift | Mobility patterns may change; older training data may become less representative |