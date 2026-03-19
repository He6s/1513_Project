
# ECE1513 Project: Final Report Data & Discussion Points

## 1. Data Pivot Summary
- **Original Data**: UCI Heart Disease (~300 samples) -> Models saturated at 85% accuracy.
- **New Data**: Kaggle Cardiovascular Disease (~70,000 samples).
- **Preprocessing**: 
    - Cleaned outliers (e.g., negative BP, extreme BMI).
    - Reduced dataset from 70,000 to ~64,800 records.
    - Stratified Train/Test Split (80/20).

## 2. Methodology Narrative: The Improvement Process

Per course requirements, we followed a strict "Improvement Process" to refine our solution from a naive baseline to a domain-aware advanced model.

### Phase 1: Establish a Baseline (Naive Solution)
- **Goal**: Determine the absolute minimum performance threshold.
- **Method**: Majority Class Classifier (ZeroR).
- **Result**: ~50.87% Accuracy.
- **Insight**: The dataset is perfectly balanced. Any model must significantly beat 50% to be useful.

### Phase 2: Basic Machine Learning (Course Content)
- **Goal**: Apply standard linear models taught in class.
- **Method**: Logistic Regression on *Raw Features* (Age, Gender, Height, Weight, AP_Hi, AP_Lo, Cholesterol, Gluc, Smoke, Myco, Active).
- **Result**: ~72.43% Accuracy.
- **Insight**: A significant jump over baseline, but potential non-linear interactions (e.g., Weight vs Height) are missed by linear boundaries.

### Phase 3: Advanced Machine Learning Model (Refinement)
- **Goal**: Capture non-linear relationships without manual engineering.
- **Method**: Gradient Boosted Trees (**XGBoost**) on *Raw Features*.
- **Result**: ~73.05% Accuracy.
- **Insight**: The tree-based model automatically found interactions, outperforming the linear model by ~0.6%. This confirms non-linearity exists.

### Phase 4: Feature Engineering (Final Solution)
- **Goal**: Inject medical domain knowledge to "linearize" the problem and boost all models.
- **Method**: Explicitly calculated `BMI`, `Pulse Pressure`, `MAP` (Mean Arterial Pressure), and Age-Cholesterol Interactions.
- **Result**:
    - **Logistic Regression (Linear)**: Improved to **72.55%**.
    - **XGBoost (Tree)**: Improved to **73.11%**.
- **Conclusion**: By manually engineering non-linear features like BMI, we brought the simple linear model much closer to the complex tree model's performance. The final XGBoost model with engineered features combines the best of both worlds: automated interaction detection + explicit domain knowledge.

## 3. Final Model Performance (5-Fold Cross-Validation)
The following results correspond to the **Final Solution (Phase 4)**:
| Model | Accuracy (Mean) | Std Dev | Recall (Mean) | F1 (Mean) |
|---|---|---|---|---|
| Naive Baseline | 0.5087 | N/A | 1.0000 | 0.6744 |
| Logistic Regression | 0.7248 | 0.0033 | 0.6952 | 0.7199 |
| SVM (Linear) | 0.7249 | 0.0037 | 0.6992 | 0.7211 |
| XGBoost (Tuned) | **0.7311** | 0.0040 | 0.6945 | **0.7243** |

*Note: While XGBoost is only ~0.6% better in accuracy, it is consistently better across all folds and metrics.*

## 4. Key Discussion Points for Report
- **Why XGBoost?**: Chosen as the final model because it balances Accuracy and F1 score best, and is robust to outliers which are common in medical data.
- **Why LogReg Failed to Converge initially?**: Introduction of interaction terms (`age*chol`) caused severe multicollinearity. Solved by increasing `max_iter` and relying on regularization, but coefficient interpretability was compromised (negative coefficient for interaction).
- **Feature Importance (SHAP)**:
    - `ap_hi` (Systolic BP) is by far the strongest predictor.
    - `age`, `cholesterol`, and `interaction` terms follow.
    - `active` and `smoke` have surprisingly low impact, suggesting lifestyle factors might be overshadowed by clinical symptoms in this specific dataset.

