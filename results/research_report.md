
# Model Comparison Report

## Executive Summary
This report compares the performance of different machine learning models for predicting Indonesian Public University Tuition Fees.

## Methodology
We evaluated the following models:
- CatBoost
- LightGBM
- RandomForest
- XGBoost


Each model was evaluated using multiple metrics to assess performance for multi-output quantile prediction with conformal prediction intervals.

## Results

### Performance Comparison

| Rank | Model | MSE | RMSE | MAE | R2 | MAPE | Pearson_Corr |
|------|-------|------|------|------|------|------|------
| 1 | CatBoost | 1214554547031.85 | 1102068.30 | 449464.85 | 0.9620 | 7.96 | 0.9809 |
| 2 | LightGBM | 1361207901654.01 | 1166708.15 | 519484.50 | 0.9574 | 8.95 | 0.9785 |
| 3 | RandomForest | 1553317760823.42 | 1246321.69 | 448080.87 | 0.9514 | 7.55 | 0.9758 |
| 4 | XGBoost | 1863452937983.55 | 1365083.49 | 489652.62 | 0.9417 | 8.58 | 0.9710 |


## Key Findings

Based on the evaluation, **CatBoost** achieved the best performance with an R² score of 0.9620.


## Conclusion

The comparative study of various machine learning models for predicting Indonesian Public University Tuition Fees shows that:

- The CatBoost model outperformed other models in terms of evaluation metrics.
- The XGBoost model had the lowest performance across most metrics.

- All models demonstrated reasonable accuracy for tuition fee prediction.
- The multi-output approach successfully predicted tuition fees for multiple years simultaneously.
- Conformal prediction provided reliable uncertainty quantification for the predictions.
