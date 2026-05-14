# Bike Rental Forecasting Project - Comprehensive Analysis

**Generated**: 14. Mai 2026  
**Project**: Daily Bike Rental Demand Forecasting (Top 20 Stations)

---

## 1. Executive Summary

### Project Scope
- **Task**: Predict daily `total_rentals` for bike-sharing top 20 stations
- **Dataset**: 31,289 daily records × 18 features (from 2020-2023)
- **Target Variable**: Single aggregated target `total_rentals`
- **Train/Val/Test Split**: 70% / 15% / 15% (chronological by `time_idx`)
- **Models Tested**: 9 different algorithms
- **Best Model**: Random Forest (Test RMSE: 30.31, R²: 0.406)

---

## 2. Model Performance Comparison

### 2.1 Complete Performance Table (Train/Val/Test)

| Model | RMSE (Train) | RMSE (Val) | RMSE (Test) | R² (Train) | R² (Val) | R² (Test) | MAE (Test) | Val Rank |
|-------|------|------|------|--------|--------|--------|--------|---------|
| **Dummy** | 37.81 | 46.10 | 48.56 | 0.000 | -0.533 | -0.523 | 38.14 | 9 |
| **Linear** | 28.23 | 32.94 | 36.72 | 0.443 | 0.217 | 0.129 | 29.40 | 8 |
| **Ridge** | 28.25 | 32.37 | 35.91 | 0.442 | 0.244 | 0.167 | 28.68 | 7 |
| **Lasso** | 28.79 | 31.13 | 33.49 | 0.420 | 0.301 | **0.275** | 26.21 | 5 |
| **DecisionTree** | 14.97 | 28.64 | 32.03 | 0.843 | 0.408 | 0.337 | 24.32 | 3 |
| **KNN** | **0.012** | 31.14 | 32.35 | **0.9998** | 0.301 | 0.324 | 24.42 | 6 |
| **Random Forest** | 17.76 | 27.28 | **30.31** | 0.779 | 0.463 | **0.406** | **22.87** | **1** |
| **Gradient Boosting** | 21.15 | **27.00** | 30.57 | 0.687 | **0.474** | 0.396 | 23.71 | **2** |
| **Neural Network** | 21.95 | 29.28 | 31.54 | 0.663 | 0.381 | 0.357 | 23.60 | 4 |

### 2.2 Key Performance Metrics Summary

**Best Model by Metric:**
- **Test RMSE**: Random Forest (30.31) - **Primary objective**
- **Validation RMSE**: Gradient Boosting (27.00)
- **Validation R²**: Gradient Boosting (0.474)
- **Test R²**: Random Forest (0.406)

**Ranking by Test Performance:**
1. 🥇 Random Forest: RMSE=30.31, MAE=22.87, R²=0.406
2. 🥈 Gradient Boosting: RMSE=30.57, MAE=23.71, R²=0.396
3. 🥉 Lasso: RMSE=33.49, MAE=26.21, R²=0.275

**Worst Performers:**
- Dummy Regressor: R² = -0.523 (predicts constant mean)
- Linear Regression: R² = 0.129 (underperforms baseline)

---

## 3. Feature Set Analysis

### 3.1 Features Used in All Models

**17 Features Total:**

| Category | Features |
|----------|----------|
| **Station** | `start_station_id` (identifier, not a predictor) |
| **Temperature** | `tempmax`, `humidity`, `sealevelpressure`, `uvindex` |
| **Precipitation** | `precip`, `precipcover`, `snow`, `snowdepth` |
| **Visibility** | `visibility`, `cloudcover`, `windspeed` |
| **Time** | `month`, `weekday`, `year`, `time_idx`, `sunset_minutes` |

**Numeric Predictors (16 features):** All except `start_station_id`

### 3.2 Feature Correlations - Critical Issues

**Highly Correlated Predictor Pairs (|r| > 0.98):**
- `solarenergy` ↔ `solarradiation` (0.9998) ❌ **REMOVED but features still exist**
- `feelslike` ↔ `temp` (0.9969)
- `feelslikemin` ↔ `tempmin` (0.9947)
- `feelslikemax` ↔ `tempmax` (0.9919)
- `daylight_minutes` ↔ `sunset_minutes` (0.9852)
- `temp` ↔ `tempmin` (0.9822)

**Features Not Reduced Despite High Correlation:**
According to [analysis/high_correlation_pairs.csv](data/processed/analysis/high_correlation_pairs.csv), 26 pairs with r > 0.9 should have been candidates for removal, but only temperature/weather redundancy was noted.

### 3.3 Feature-Target Correlation (Top Predictors)

Based on [analysis/predictor_target_correlations.csv](data/processed/analysis/predictor_target_correlations.csv):
- **Positive**: `tempmax` (0.36), `temp` (0.34), `uvindex` (0.29)
- **Negative**: `precipcover` (-0.18), `precip` (-0.13)
- **Weak Signals**: Most features have |r| < 0.3 with target

**Interpretation**: Single features are weak predictors → ensemble methods (RF, GB) needed.

---

## 4. Data Quality Issues

### 4.1 Missing Values - CRITICAL ISSUES

| Feature | Missing Count | % Missing | Status |
|---------|---------|---------|---------|
| `severerisk` | **12,030** | **38.4%** | ❌ **NOT HANDLED** |
| All other 16 features | 0 | 0% | ✓ OK |

**Problem Severity**: 🔴 **CRITICAL**
- `severerisk` with 38% missing values is silently dropped or causes issues
- No explicit null handling in preprocessing pipeline
- May cause silent failures in model training

### 4.2 Data Leakage Risks

**Station-Time Aggregation Problem:**
1. Raw data: Daily rentals per start station
2. Preprocessing combines features across all stations
3. Train/val/test split by `time_idx` assumes no station-wise leakage
4. **Risk**: Features might include information from multiple dates/stations

**Time Series Assumptions:**
- Chronological split correctly prevents future information leakage
- BUT: `time_idx` appears to be numeric ID, not an actual timestamp
- Potential issue: Non-sequential time values could break chronological split

### 4.3 Data Imbalance

**Target Variable Distribution:**
- Mean rentals: ~65.7 (from intercept ~64.66 across models)
- Range: 0 to potentially very high
- Outliers: Some visible (e.g., row 36 shows 179 rentals)

**No investigation of**:
- Outlier distribution
- Seasonal patterns beyond month/weekday encoding
- Station-specific demand patterns

---

## 5. Implementation Problems

### 5.1 Critical Issues 🔴

#### 1. **KNN Severe Overfitting**
- **Train R²**: 0.9998 (virtually perfect fit)
- **Validation R²**: 0.301 (catastrophic drop)
- **Train MAE**: 0.012 vs **Validation MAE**: 23.5
- **Root Cause**: Scaling applied, but k=25 too small for 31,289 samples
- **Impact**: Model is useless for production despite apparent performance
- **Location**: [modelling/05_knn_regressor/train_knn.py](modelling/05_knn_regressor/train_knn.py)

#### 2. **Missing Values Not Handled**
- `severerisk` has 38% missing values
- No explicit handling in [modelling/common/preprocessing.py](modelling/common/preprocessing.py)
- Models may silently fail or use default imputation
- **Impact**: Unknown behavior with incomplete data
- **Fix Required**: Add explicit NaN handling/imputation

#### 3. **No Feature Selection Despite Multicollinearity**
- 26 feature pairs have correlation > 0.9
- Analysis identifies these but no removal implemented
- Linear models suffer from multicollinearity
- Ridge/Lasso somewhat mitigate but suboptimal
- **Impact**: Inflated coefficients, poor interpretability
- **Location**: [src/scripts/03_analyze_feature_correlations.py](src/scripts/03_analyze_feature_correlations.py) (analysis done but not applied)

### 5.2 High Priority Issues 🟠

#### 4. **Hyperparameter Tuning Not Saved**
- Only final metrics saved in JSON
- No search grid history, best parameter evolution, or diagnostic plots
- Cannot reproduce tuning decisions or debug model selection
- **Locations**: 
  - [modelling/01_linear_regression/train_linear_regression.py](modelling/01_linear_regression/train_linear_regression.py)
  - [modelling/02_ridge_regression/train_ridge.py](modelling/02_ridge_regression/train_ridge.py)
  - [modelling/06_random_forest/train_random_forest.py](modelling/06_random_forest/train_random_forest.py)

#### 5. **KNN Prediction Time Inefficiency**
- Predict time: **0.832 seconds** for 4,755 test samples
- Other models: < 0.2s
- **Cause**: k-d tree search for k=25 neighbors
- **Impact**: Not suitable for production/real-time forecasting

#### 6. **Neural Network Broken Function Reference**
- Code references `plot_feature_importance()` that doesn't exist
- Will cause runtime error if plotting is executed
- **Location**: [modelling/08_neural_network/train_neural_network.py](modelling/08_neural_network/train_neural_network.py) line references plotting functions

#### 7. **Incomplete Hyperparameter Search Results**
- **Ridge**: Only one alpha tested (1000.0) - why? No explanation
- **Lasso**: Only one alpha tested (2.0)
- **Random Forest**: Parameter grid exists but best values unclear
- **Gradient Boosting**: Limited hyperparameter exploration
- **Impact**: May not have found truly optimal parameters

### 5.3 Medium Priority Issues 🟡

#### 8. **No Cross-Validation**
- Simple train/val/test split, no k-fold CV
- Single split results may be unstable
- Cannot estimate confidence intervals on metrics
- **Location**: [modelling/common/split.py](modelling/common/split.py) - only chronological_split, no CV

#### 9. **Station Information Not Used**
- `start_station_id` kept but not encoded in modeling
- Each station has unique demand patterns (should be one-hot encoded)
- Random Forest cannot fully capture per-station effects
- **Impact**: Leaves significant predictive information unused
- **Location**: [modelling/common/preprocessing.py](modelling/common/preprocessing.py) line 9 excludes non-numeric

#### 10. **No Residual Analysis or Diagnostics**
- Predictions saved but no systematic residual analysis
- Cannot detect systematic errors, heteroscedasticity, or temporal patterns
- Plotting functions exist but not integrated into training
- **Evidence**: Plotting functions in [modelling/common/plotting.py](modelling/common/plotting.py) not called during training

#### 11. **Inconsistent Scaling**
- Linear/Ridge/Lasso use StandardScaler
- Tree-based models use unscaled data (correct)
- BUT: KNN uses scaled features from preprocessing
- **Location**: [modelling/common/preprocessing.py](modelling/common/preprocessing.py) - all models call scale_features but trees don't use it

#### 12. **No Temporal Validation Strategy**
- `time_idx` is numeric, unclear if truly chronological
- No explicit time-series cross-validation (e.g., expanding window)
- Single split may not capture all seasonal patterns
- **Risk**: Models may have implicitly learned test set patterns if time_idx is not perfectly ordered

### 5.4 Code Structure Issues 🟡

#### 13. **Duplicate Utility Functions**
- `_json_converter()` and `ensure_dir()` repeated in multiple files
- DRY principle violated
- **Locations**: 
  - [modelling/common/utils.py](modelling/common/utils.py)
  - [modelling/99_model_comparison/model_comparison.py](modelling/99_model_comparison/model_comparison.py)

#### 14. **Inconsistent Metric Naming**
- Mixed use of `train_mae`, `validation_mae`, `test_mae` vs `val_mae`, `test_mae`
- JSON column names vary between models
- Complicates comparison logic
- **Location**: [modelling/common/metrics.py](modelling/common/metrics.py) line 19-26

#### 15. **Hard-coded Magic Numbers**
- Split ratios (0.70, 0.15, 0.15) in config
- Random seed (42) not justified
- Scaling method (StandardScaler) not configurable
- **Location**: [modelling/common/config.py](modelling/common/config.py)

---

## 6. Data Quality Deep Dive

### 6.1 Preprocessing Pipeline Summary

**Pipeline Steps:**
1. ✓ Load raw data (48 columns)
2. ✓ Identify top 20 start stations by usage frequency
3. ✓ Aggregate to daily level by station
4. ✓ Merge weather data
5. ✓ Analyze correlations (identify 26 redundant pairs)
6. ✓ Remove "obvious" duplicates (solarenergy, solarradiation)
7. ✓ Remove count columns (member_count, casual_count, etc.)
8. ✓ Remove text columns (station_name)
9. ✗ **NO NULL HANDLING** ← Problem
10. ✓ Final dataset: 31,289 × 18

**Problematic Decisions:**
- Removed 32 columns without domain validation
- Kept `severerisk` despite 38% NAs
- No imputation strategy documented
- Single target (total_rentals) discards potentially useful multi-target info

### 6.2 Data Characteristics

**31,289 daily records across:**
- 20 bike-sharing stations
- 5 years (2020-2023)
- ~314 days per station on average
- Monthly and weekday temporal features

**Feature Statistics:**
- All numeric (int/float64)
- Most complete (0 NAs except severerisk)
- Range varies widely (e.g., uvindex 0-11, tempmax -10 to 40+°C)
- Properly scaled with StandardScaler before linear models

---

## 7. Problem Priority Matrix

| Severity | Problem | Impact | Location | Fix Complexity |
|----------|---------|--------|----------|-----------------|
| 🔴 CRITICAL | Missing `severerisk` (38% NAs) | Model corruption | preprocessing.py | Medium |
| 🔴 CRITICAL | KNN overfitting (R²: 0.9998→0.30) | Unreliable production model | train_knn.py | High |
| 🔴 CRITICAL | No feature selection despite multicollinearity | Linear model instability | scripts/03_analyze.py | Medium |
| 🟠 HIGH | Hyperparameter tuning not logged | Cannot debug/reproduce | train_*.py files | Low |
| 🟠 HIGH | Neural Network plotting function missing | Runtime error risk | train_neural_network.py | Low |
| 🟠 HIGH | Station ID not encoded | Information loss | preprocessing.py | Medium |
| 🟡 MEDIUM | No cross-validation | Results may be unstable | split.py | Medium |
| 🟡 MEDIUM | KNN prediction latency (0.83s) | Not production-ready | train_knn.py | Low |
| 🟡 MEDIUM | Inconsistent scaling logic | Subtle bugs possible | preprocessing.py | Low |
| 🟡 MEDIUM | Duplicate utility functions | Code maintainability | multiple files | Low |

---

## 8. Recommendations

### Immediate (Critical Path)
1. **Add explicit null handling** for `severerisk` - impute or drop column
2. **Fix KNN overfitting** - increase k value (e.g., k=50-100) or use cross-validation
3. **Implement feature selection** - remove correlated features before linear models
4. **Log hyperparameter search** - save search history, not just final metrics

### Short Term (1-2 days)
5. **Add per-station features** - one-hot encode station ID or use station embeddings
6. **Implement time-series cross-validation** - expanding window validation
7. **Add residual diagnostics** - plot residuals over time, by station
8. **Fix Neural Network code** - either implement feature importance or remove plotting

### Medium Term (refactoring)
9. **Consolidate utility functions** - move duplicates to utils.py
10. **Add config validation** - check for impossible split ratios
11. **Document feature selection rationale** - why was severerisk kept?
12. **Add model reproducibility** - save random seeds, data hashes

### Optional Improvements
13. Multi-target modeling (casual vs member rentals)
14. Exogenous variables (holidays, events)
15. AutoML/Bayesian optimization for hyperparameter search
16. Ensemble of best models (Random Forest + Gradient Boosting)

---

## 9. File Structure Review

✓ **Well Organized:**
- Clear separation: data processing / modeling / comparison
- Consistent naming across model folders
- Common utility module for shared code
- Results saved in consistent JSON/CSV format

⚠️ **Could Improve:**
- No documentation strings in config.py
- No README in modelling/ folder explaining train script usage
- No requirements.txt visible (dependencies not listed)
- No logging system (only print statements likely)

---

## Appendix: Raw Metrics Data

### All Models - Validation Performance
```
Gradient Boosting: RMSE=27.00 (Best) | MAE=20.53 | R²=0.474
Random Forest:     RMSE=27.28       | MAE=20.40 | R²=0.463
Decision Tree:     RMSE=28.64       | MAE=21.33 | R²=0.408
Lasso:             RMSE=31.13       | MAE=24.04 | R²=0.301
KNN:               RMSE=31.14       | MAE=23.50 | R²=0.301
Neural Network:    RMSE=29.28       | MAE=22.20 | R²=0.381
Ridge:             RMSE=32.37       | MAE=25.57 | R²=0.244
Linear:            RMSE=32.94       | MAE=26.14 | R²=0.217
Dummy:             RMSE=46.10       | MAE=36.99 | R²=-0.533
```

### All Models - Test Performance
```
Random Forest:     RMSE=30.31 (Best) | MAE=22.87 | R²=0.406
Gradient Boosting: RMSE=30.57        | MAE=23.71 | R²=0.396
Lasso:             RMSE=33.49        | MAE=26.21 | R²=0.275
Neural Network:    RMSE=31.54        | MAE=23.60 | R²=0.357
Decision Tree:     RMSE=32.03        | MAE=24.32 | R²=0.337
KNN:               RMSE=32.35        | MAE=24.42 | R²=0.324
Ridge:             RMSE=35.91        | MAE=28.68 | R²=0.167
Linear:            RMSE=36.72        | MAE=29.40 | R²=0.129
Dummy:             RMSE=48.56        | MAE=38.14 | R²=-0.523
```

---

**Analysis Completed**: 2026-05-14  
**Status**: Ready for remediation planning
