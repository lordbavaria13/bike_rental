# Time Error Insight Report

This report is generated from test predictions and temporal features.

It summarizes whether errors differ by weekday, month position, and year position.

## Gradient Boosting

- Weekday: lowest MAE on/at Fri (14.79), highest MAE on/at Sat (20.92).
- Month position: lowest MAE on/at Beginning (16.14), highest MAE on/at End (17.69).
- Year position: lowest MAE on/at Beginning (16.60), highest MAE on/at Middle (17.21).

## Neural Network

- Weekday: lowest MAE on/at Mon (16.67), highest MAE on/at Sun (21.98).
- Month position: lowest MAE on/at Beginning (18.15), highest MAE on/at End (20.23).
- Year position: lowest MAE on/at Beginning (18.22), highest MAE on/at Middle (19.50).

## Lasso

- Weekday: lowest MAE on/at Fri (15.24), highest MAE on/at Sat (23.98).
- Month position: lowest MAE on/at Beginning (16.44), highest MAE on/at Middle (18.72).
- Year position: lowest MAE on/at Beginning (17.17), highest MAE on/at Middle (18.33).

## How to use this in Q&A

- Use the weekday plot to discuss whether weekday or weekend demand is harder to predict.
- Use the month-position plot to discuss whether beginning or end of month creates systematic errors.
- Use the year-position plot to discuss seasonal and end-of-year effects.
- These plots are diagnostic and descriptive, not causal proof.
