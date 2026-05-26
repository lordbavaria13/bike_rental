# Detailed Time Error Insight Report

This report is generated from test predictions and reconstructed calendar information.

It evaluates whether errors differ by individual day of month and individual month of year.

## Gradient Boosting

- Best day of month by MAE: day 9 with MAE 13.63.
- Worst day of month by MAE: day 16 with MAE 21.70.
- Best month by MAE: Feb with MAE 13.65.
- Worst month by MAE: Mar with MAE 18.90.

## Neural Network

- Best day of month by MAE: day 12 with MAE 14.29.
- Worst day of month by MAE: day 28 with MAE 22.74.
- Best month by MAE: Feb with MAE 17.28.
- Worst month by MAE: Jul with MAE 21.29.

## Lasso

- Best day of month by MAE: day 7 with MAE 14.61.
- Worst day of month by MAE: day 16 with MAE 23.91.
- Best month by MAE: Feb with MAE 13.36.
- Worst month by MAE: Mar with MAE 19.89.

## How to use this in Q&A

- Use day-of-month plots to discuss whether beginning, middle, or end of month contains harder cases.
- Use month-of-year plots to discuss seasonal effects more precisely than the earlier Beginning/Middle/End-of-year plot.
- Use bias plots to explain whether a model systematically overpredicts or underpredicts in specific time periods.
- These diagnostics are descriptive and should not be interpreted as causal proof.
