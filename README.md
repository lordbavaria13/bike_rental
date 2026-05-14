# Bike Rental Demand Forecasting Project

## Project goal

The project predicts the daily number of bike rentals.

The final target variable is:

- `total_rentals`

The work started with raw bike rental and weather data.  
After that, the data was reduced step by step into one clean modelling table.  
Then several regression models were trained and compared under the same setup.

---

## Current project structure

```text
bike_rental/
├── data/
│   └── processed/
├── src/
│   └── scripts/
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
│   └── 99_model_comparison/
```

---

## What the code does overall

The code currently does four main things:

1. **Builds a daily station-level dataset**
2. **Analyzes correlations and reduces features**
3. **Trains several regression models with the same split logic**
4. **Compares all trained models with common metrics and plots**

---

## Data processing pipeline

### `src/scripts/02_build_top20_daily_dataset.py`

This script creates the first processed dataset.

It does the following:

- finds the **top 20 start stations**
- aggregates the raw rental data to **daily level**
- merges station information
- merges weather information
- saves the processed daily dataset

Main outputs:

- `data/processed/top_20_stations.csv`
- `data/processed/daily_rentals_top20.csv.gz`

This file is still broad.  
It still contains many count variables, weather variables, time variables, station names, and date-related columns.

---

### `src/scripts/03_analyze_feature_correlations.py`

This script analyzes redundancy in the processed dataset.

It does the following:

- loads the processed daily dataset
- separates numeric predictors and target-related columns
- computes a **correlation matrix**
- creates a **heatmap**
- detects highly correlated predictor pairs
- recommends which columns can be dropped

Main idea:

- if two predictors contain almost the same information, one of them can be removed
- this makes the dataset simpler
- it also reduces multicollinearity for linear models

Typical examples from the analysis were:

- `solarenergy` vs `solarradiation`
- `feelslike` vs `temp`
- `tempmin` vs `feelslikemin`
- `feelslikemax` vs `tempmax`

Main output folder:

- `data/processed/analysis/`

This folder contains the correlation outputs and the heatmap used for feature selection.

---

### `src/scripts/04_build_reduced_feature_dataset.py`

This script builds the final modelling dataset.

This is the most important preprocessing script.

It does the following:

- loads the larger processed daily dataset
- removes columns that are not needed for modelling
- removes leakage-like or redundant count columns
- keeps only one target:
  - `total_rentals`
- keeps only numeric predictor columns
- removes station name text columns
- keeps `start_station_id` as station identifier
- keeps selected weather and time variables
- saves a clean reduced dataset for modelling
- saves a station mapping file

Final modelling dataset:

- `data/processed/daily_rentals_top20_reduced.csv`
- `data/processed/daily_rentals_top20_reduced.csv.gz`

Additional outputs:

- `data/processed/station_id_name_mapping.csv`
- `data/processed/daily_rentals_top20_removed_columns.csv`
- `data/processed/daily_rentals_top20_reduced_summary.csv`

### Final feature set

The final reduced table contains:

- `start_station_id`
- `tempmax`
- `humidity`
- `precip`
- `precipcover`
- `cloudcover`
- `windspeed`
- `visibility`
- `sealevelpressure`
- `uvindex`
- `sunset_minutes`
- `month`
- `weekday`
- `year`
- `time_idx`
- `snow`
- `snowdepth`

Target column:

- `total_rentals`

### Why this reduction was done

The reduction was done to create a clean regression table.

Main decisions:

- only one target was kept, because predicting several count targets at once would make the setup less clean
- sub-counts by bike type and member/casual were removed
- non-numeric text columns were removed
- the original raw date column was removed
- only a compact set of weather and simple time variables was kept
- highly correlated variables were reduced based on the heatmap and correlation analysis

---

## `modelling/common/`

This folder contains shared helper code.

All model scripts use these files.  
The goal is that every model is trained under the same setup.

### `modelling/common/config.py`

Central configuration file.

Contains for example:

- path to the reduced dataset
- target column name
- time column name
- station column name
- train / validation / test ratios
- random seed
- default plot settings

Important constants include:

- `DATA_PATH`
- `TARGET_COL`
- `TIME_COL`
- `STATION_COL`

---

### `modelling/common/metrics.py`

Contains common regression metrics.

Main functions:

- `safe_mape(...)`
- `compute_regression_metrics(...)`

The following metrics are computed for train, validation, and test:

- MAE
- RMSE
- R²
- Median Absolute Error
- Explained Variance
- MAPE

---

### `modelling/common/plotting.py`

Contains shared plotting functions.

Typical plots include:

- actual vs predicted
- residual histogram
- residuals vs predicted
- error over time
- feature importance

This ensures that all model result plots use the same style.

---

### `modelling/common/preprocessing.py`

Contains helper functions for loading and preparing data.

Main functions:

- `load_dataset(...)`
- `get_numeric_feature_columns(...)`
- `split_X_y(...)`
- `scale_features(...)`

Important note:

- scaling is mainly needed for models like Ridge, Lasso, and KNN
- tree-based models do not need feature scaling

---

### `modelling/common/split.py`

Contains the chronological split logic.

Main function:

- `chronological_split(...)`

This function:

- sorts by the time variable
- creates train, validation, and test splits in time order
- avoids random shuffling

This is important because the task is time-based demand prediction.

---

### `modelling/common/utils.py`

Contains utility functions.

Main functions:

- `ensure_dir(...)`
- `ensure_dirs(...)`
- `save_json(...)`
- `save_dataframe(...)`

This file handles directory creation and saving outputs in a consistent way.

---

## Model folders

Each model folder follows the same basic structure.

Typical structure:

```text
model_folder/
├── train_....py
├── results/
│   ├── metrics.csv
│   ├── metrics.json
│   ├── predictions.csv
│   ├── hyperparameter_search.csv   # if used
│   ├── feature_importance.csv      # if available
│   └── plots/
├── model/
│   ├── model file
│   ├── scaler file                 # only if needed
│   └── model_info.json
```

---

## `modelling/00_dummy_regressor/`

### `train_dummy.py`

This is the baseline model.

It predicts a simple constant value based on the training data mean.

Purpose:

- gives the simplest possible benchmark
- shows whether more advanced models really improve over a naive prediction

Outputs include:

- metrics
- predictions
- standard result plots

This model is important because every real model should beat this baseline.

---

## `modelling/01_linear_regression/`

### `train_linear_regression.py`

This script trains a standard linear regression model.

It does the following:

- loads the reduced dataset
- uses chronological splitting
- selects numeric features
- scales the features
- trains linear regression
- saves coefficients
- saves predictions and plots
- evaluates performance on train, validation, and test

Why this model is included:

- it is the simplest real regression model
- it is interpretable
- it shows how far a linear relationship can already explain bike rental demand

---

## `modelling/02_ridge_regression/`

### `train_ridge.py`

This script trains Ridge Regression.

It does the following:

- uses the same dataset and split as the other models
- scales the features
- searches over different `alpha` values
- selects the best parameter on the validation set
- trains the final model
- saves metrics, predictions, coefficients, and plots

Why Ridge was used:

- Ridge adds L2 regularization
- this helps when predictors are correlated
- it often improves stability over plain linear regression

This is especially relevant here because weather and time variables can still be related to each other.

---

## `modelling/03_lasso_regression/`

### `train_lasso.py`

This script trains Lasso Regression.

It is similar to Ridge, but uses L1 regularization.

It does the following:

- scales features
- searches over `alpha`
- picks the best model on the validation set
- saves metrics and plots

Why Lasso was used:

- Lasso can shrink some coefficients to zero
- this gives a simpler model
- it can act like automatic feature selection

This is useful for testing whether only a smaller subset of the final features is really needed.

---

## `modelling/04_decision_tree/`

### `train_decision_tree.py`

This script trains a decision tree regressor.

It does the following:

- uses the reduced numeric dataset
- uses the chronological split
- tests several tree settings
- chooses the best setting by validation performance
- trains the final tree
- saves predictions, metrics, feature importances, and plots

Why this model is included:

- it can model nonlinear relationships
- it can capture interaction effects
- it does not require feature scaling

Weak point:

- decision trees can overfit easily
- this is why `max_depth`, `min_samples_leaf`, and `min_samples_split` are tuned

---

## `modelling/05_knn_regressor/`

### `train_knn.py`

This script trains a K-Nearest Neighbors regressor.

It does the following:

- scales the features
- tests several values for:
  - `n_neighbors`
  - `weights`
  - distance norm `p`
- chooses the best setup on the validation set
- saves predictions, metrics, and plots

Why this model is included:

- KNN is a simple non-parametric model
- it predicts based on similar past observations
- it gives a different modeling idea than linear or tree-based methods

Important note:

- KNN depends strongly on feature scaling
- this is why standardized inputs are required

---

## `modelling/06_random_forest/`

### `train_random_forest.py`

This script trains a random forest regressor.

It does the following:

- tests several settings for:
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `min_samples_split`
  - `max_features`
- chooses the best setup on the validation set
- trains the final model
- saves metrics, predictions, feature importances, and plots

Why this model is included:

- random forest reduces overfitting compared to a single tree
- it can capture nonlinear effects
- it can handle interactions between variables
- it often performs strongly on structured tabular data

---

## `modelling/07_gradient_boosting/`

### `train_gradient_boosting.py`

This script trains a gradient boosting regressor.

It does the following:

- tests several settings for:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `min_samples_leaf`
  - `subsample`
- selects the best configuration on the validation set
- trains the final model
- saves metrics, predictions, feature importance, and plots

Why this model is included:

- boosting often performs very well on tabular regression tasks
- it builds many weak trees step by step
- later trees correct earlier errors
- it is usually more flexible than a single tree and often stronger than simpler models

---

## `modelling/99_model_comparison/`

### `model_comparison.py`

This script compares all finished model runs.

It does the following:

- searches all model folders for `results/metrics.csv`
- skips folders without finished results
- combines all available model metrics into one comparison table
- builds validation and test rankings
- creates many comparison plots
- highlights the best models
- saves summary files

Main outputs include:

- `all_model_metrics.csv`
- `model_comparison_compact.csv`
- `model_ranking_validation_rmse.csv`
- `model_ranking_test_rmse.csv`
- `comparison_summary.json`

Plots include:

- MAE comparison
- RMSE comparison
- Median AE comparison
- MAPE comparison
- R² comparison
- Explained Variance comparison
- fit time comparison
- predict time comparison
- validation RMSE ranking
- test RMSE ranking
- validation rank sum
- test rank sum
- one large overview dashboard

This script is the final comparison step of the current pipeline.

---

## Current evaluation setup

All models are evaluated with the same general strategy.

### Data split

The split is chronological:

- 70% train
- 15% validation
- 15% test

Why this is done:

- the task is time-based
- random splitting would mix past and future
- that would make the evaluation less realistic

### Metrics

The same metrics are used for every model:

- **MAE** = Mean Absolute Error
- **RMSE** = Root Mean Squared Error
- **Median Absolute Error**
- **MAPE** = Mean Absolute Percentage Error
- **R²**
- **Explained Variance**

Why several metrics are used:

- MAE is easy to interpret
- RMSE punishes large errors more strongly
- Median AE is more robust to outliers
- MAPE shows relative error
- R² and Explained Variance show how much structure is captured by the model

---

## What is already implemented

Up to now, the codebase includes:

- a preprocessing pipeline
- correlation analysis
- a reduced final modelling dataset
- common utilities for all models
- baseline model
- linear models
- nonlinear models
- model comparison pipeline
- automatic saving of metrics, plots, predictions, and model files

---

## Main limitations of the current code and setup

The current pipeline is already structured, but it still has some weaknesses.

### 1. Station ID is still numeric

`start_station_id` is currently treated like a numeric variable.

This is practical, but not ideal.

A station ID is really a category, not a true numeric quantity.  
This may limit especially linear models.

### 2. Time features are still simple

The current time variables are simple:

- `month`
- `weekday`
- `year`
- `time_idx`
- `sunset_minutes`

This is clear and compact, but still basic.

### 3. No lag features are included

The final modelling table does not yet include previous demand values.

That means the models only use same-day explanatory features.

### 4. Model quality is still moderate

The project already compares many models, but the predictions are still not very strong.

This means that:
- either the current features are not rich enough
- or more tuning is needed
- or both

---

## Recommended next steps

The next useful steps are:

1. compare all completed models carefully with `99_model_comparison`
2. identify the strongest current baseline
3. inspect residual plots by model
4. improve feature design
5. retune the strongest models
6. then run the full comparison again

Examples of reasonable next improvements:

- better encoding of station information
- better representation of time patterns
- more systematic hyperparameter tuning
- learning curve analysis
- bias-variance discussion based on results
- closer residual analysis

---

## How to run the main scripts

Always run model scripts from the project root:

```bash
source .venv/bin/activate
```

Then for example:

```bash
.venv/bin/python -m modelling.00_dummy_regressor.train_dummy
.venv/bin/python -m modelling.01_linear_regression.train_linear_regression
.venv/bin/python -m modelling.02_ridge_regression.train_ridge
.venv/bin/python -m modelling.03_lasso_regression.train_lasso
.venv/bin/python -m modelling.04_decision_tree.train_decision_tree
.venv/bin/python -m modelling.05_knn_regressor.train_knn
.venv/bin/python -m modelling.06_random_forest.train_random_forest
.venv/bin/python -m modelling.07_gradient_boosting.train_gradient_boosting
.venv/bin/python -m modelling.99_model_comparison.model_comparison
```

Important:

- do **not** run the model scripts from inside the Python interpreter (`>>>`)
- do **not** run them directly by file path if they import `modelling.common...`
- use the `-m` form from the project root

---

## Short summary

The current codebase already covers the full workflow from processed data to model comparison.

The workflow is:

1. build the daily top-20 station dataset
2. analyze correlation structure
3. reduce the data to one clean modelling table
4. train several regression models under the same split logic
5. compare all models with common metrics and plots

This means the project is already set up for systematic model testing and later improvement.
