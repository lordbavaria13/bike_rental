# Weather Error Insight Report

This report is generated from test predictions merged with processed weather features.

Weather categories are diagnostic bins, not original dataset labels.

## Gradient Boosting

- Favorable weather MAE: 16.78
- Difficult weather MAE: 17.30
- Difference difficult minus favorable: 0.53
- Mean residual in difficult weather: -6.04 (underprediction)

## Neural Network

- Favorable weather MAE: 19.06
- Difficult weather MAE: 19.01
- Difference difficult minus favorable: -0.05
- Mean residual in difficult weather: -6.38 (underprediction)

## Lasso

- Favorable weather MAE: 16.58
- Difficult weather MAE: 18.61
- Difference difficult minus favorable: 2.04
- Mean residual in difficult weather: 4.29 (overprediction)

## How to use this in Q&A

- Use MAE by weather condition to discuss whether bad weather increases errors.
- Use bias by weather condition to discuss systematic over- or underprediction.
- Use temperature, precipitation, wind, and humidity plots to identify specific weak spots.
- Use correlation diagnostics only as descriptive evidence, not as causal proof.
