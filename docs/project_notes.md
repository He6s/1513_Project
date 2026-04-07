# Project Notes

## Final project summary

This project predicts cardiovascular disease using the Kaggle cardiovascular disease dataset.

## Pipeline
- Data cleaning
- Feature engineering
- Logistic regression
- Linear SVM
- XGBoost
- 10-seed evaluation
- SHAP explainability

## Engineered features
- BMI
- Pulse pressure
- Mean arterial pressure
- Age-cholesterol interaction

## Final hold-out results averaged over 10 seeds
- Baseline: 50.88% ± 0.00%
- Logistic Regression: 72.11% ± 0.31%
- SVM linear: 72.07% ± 0.33%
- XGBoost: 72.94% ± 0.28%

## Final repeated cross-validation results
- Logistic Regression: accuracy 0.7246, F1 0.7198
- SVM linear: accuracy 0.7242, F1 0.7204
- XGBoost: accuracy 0.7313, F1 0.7246

## Main takeaway

All learned models clearly outperform the naive baseline. XGBoost performs best overall, but the gain over the linear models is modest after preprocessing and feature engineering.
