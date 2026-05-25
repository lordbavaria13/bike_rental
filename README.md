# Bike Rental Demand Forecasting Project

## Project Goal

The project predicts the daily number of bike rentals per station.

The final target variable is `total_rentals`.

The work started with raw bike rental and weather data.
After that, the data was reduced step by step into one clean modelling
table with one-hot encoded station identity and station-specific lag
features. Then several regression models were trained and compared
under the same setup.

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

```text
bike_rental/
├── data/
│   └── processed/
├── src/
│   └── scripts/
│       ├── 02_build_top20_daily_dataset.py
│       ├── 03_analyze_feature_correlations.py
│       ├── 04_build_reduced_feature_dataset.py
│       └── 05_create_encoded_dataset.py
├── modelling/
│   ├── common/
│   ├── 00_dummy_regressor/
│   ├── 01_linear_regression/
│   ├── 02_ridge_regression/
│   ├── 03_lasso_regression/
│   ├── 04_decision_tree/
│   ├── 05_knn_regressor/
│   ├── 06_random_forest/
│   ├── 07_gradient_boosting/
│   ├── 08_neural_network/
│   └── 99_model_comparison/
├── run_pipeline.py
└── requirements.txt
```

---

## How to Run the Full Pipeline

The easiest way to run everything is with `run_pipeline.py`:

```bash
# Activate virtual environment
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

# Run the full pipeline from the project root
python run_pipeline.py
```

Individual steps can be enabled or disabled by changing the flags
at the top of `run_pipeline.py` (e.g. `RUN_07_GRADIENT_BOOSTING = 1`).

To run a single model manually:

```bash
python -m modelling.00_dummy_regressor.train_dummy --experiment all
python -m modelling.01_linear_regression.train_linear_regression --experiment all
python -m modelling.02_ridge_regression.train_ridge --experiment all
python -m modelling.03_lasso_regression.train_lasso --experiment all
python -m modelling.04_decision_tree.train_decision_tree --experiment all
python -m modelling.05_knn_regressor.train_knn --experiment all
python -m modelling.06_random_forest.train_random_forest --experiment all
python -m modelling.07_gradient_boosting.train_gradient_boosting --experiment all
python -m modelling.08_neural_network.train_neural_network --experiment all
python -m modelling.99_model_comparison.model_comparison
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