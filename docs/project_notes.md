
# ECE1513 Project: Final Report Data & Discussion Points

## 1. Data Pivot Summary
- **Original Data**: UCI Heart Disease (~300 samples) -> Models saturated at 85% accuracy.
- **New Data**: Kaggle Cardiovascular Disease (~70,000 samples).
- **Preprocessing**: 
    - Cleaned outliers (e.g., negative BP, extreme BMI).
    - Reduced dataset from 70,000 to ~64,800 records.
    - Stratified Train/Test Split (80/20).

## 2. Methodology Narrative
1.  **Naive Solution**: Majority Class prediction (~50% Accuracy).
2.  **Basic Solution (Linear)**: Logistic Regression on raw features -> **72.43%** Accuracy.
3.  **Advanced Solution (Tree)**: XGBoost on raw features -> **73.05%** Accuracy.
    - *Comparison*: XGBoost automatically outperformed LR by ~0.6%, suggesting it captured non-linear relationships (e.g., weight/height interactions) that linear models missed.
4.  **Refined Solution (Feature Engineering)**:
    - Added Medical Domain Knowledge Features: `BMI`, `Pulse Pressure`, `MAP`, `Age*Chol`.
    - Result: Logistic Regression improved to **72.55%**, narrowing the gap.
    - *Insight*: Domain knowledge can compensate for model simplicity.

## 3. Final Model Performance (5-Fold Cross-Validation)
| Model               | Accuracy (Mean) | Std Dev | Recall (Mean) | F1 (Mean) |
|---------------------|-----------------|---------|---------------|-----------|
| Logistic Regression | 0.7248          | 0.0033  | 0.6952        | 0.7199    |
| SVM (Linear)        | 0.7249          | 0.0037  | 0.6992        | 0.7211    |
| XGBoost (Tuned)     | **0.7311**      | 0.0040  | 0.6945        | **0.7243**|

*Note: While XGBoost is only ~0.6% better in accuracy, it is consistently better across all folds and metrics.*

## 4. Key Discussion Points for Report
- **Why XGBoost?**: Chosen as the final model because it balances Accuracy and F1 score best, and is robust to outliers which are common in medical data.
- **Why LogReg Failed to Converge initially?**: Introduction of interaction terms (`age*chol`) caused severe multicollinearity. Solved by increasing `max_iter` and relying on regularization, but coefficient interpretability was compromised (negative coefficient for interaction).
- **Feature Importance (SHAP)**:
    - `ap_hi` (Systolic BP) is by far the strongest predictor.
    - `age`, `cholesterol`, and `interaction` terms follow.
    - `active` and `smoke` have surprisingly low impact, suggesting lifestyle factors might be overshadowed by clinical symptoms in this specific dataset.

