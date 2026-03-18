# Teammate Handoff - Final State

## Project Status: COMPLETE
The project has successfully pivoted to the Kaggle Cardiovascular Disease dataset (~70k records). All planned models (Logistic Regression, SVM, XGBoost) have been implemented, tuned, and evaluated.

## Key Outcomes
- **Best Model**: Tuned XGBoost (Acc: ~73.1%, F1: ~0.724).
- **Feature Engineering**: Added BMI, MAP, Pulse Pressure, and Age*Chol interactions which significantly boosted linear model performance.
- **Explainability**: SHAP plots generated (`results/plots/shap_summary.png`) confirm Systolic BP (`ap_hi`) as the top predictor.

## Files Description
- **Data**: 
    - `data/cardio_train.csv`: Raw input.
    - `data/dataset_clean.csv`: Preprocessed with new features.
- **Scripts**:
    - `src/preprocess.py`: Handles outliers and feature creation.
    - `src/evaluate_cv.py`: Main validation script (5-Fold CV).
    - `src/evaluate_comparison.py`: Generates ROC/CM plots.
    - `src/explainability.py`: Generates SHAP explanations.

## Running the Final Report Code
1. Ensure `data/cardio_train.csv` exists.
2. Run `python src/preprocess.py` to regenerate `dataset_clean.csv` and splits.
3. Run `python src/evaluate_cv.py` to get the final 5-fold CV metrics.
4. Run `python src/explainability.py` for SHAP plots.

## Known Issues / Notes
- **Logistic Regression Convergence**: Required `max_iter=5000` due to interaction terms.
- **XGBoost SHAP**: Uses `PermutationExplainer` because the native TreeExplainer had version conflicts with `base_score`. It is slower but accurate.


## Shared evaluation helper
A shared evaluation file is now available:

- `src/evaluate.py`

It computes these metrics:
- accuracy
- precision
- recall
- F1 score

Teammates should reuse this file so all models are evaluated consistently.
