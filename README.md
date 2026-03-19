# ECE1513 Project: Cardiovascular Disease Prediction

An end-to-end machine learning pipeline for binary classification of cardiovascular disease using the Kaggle dataset (~70,000 records).

## Project Status
**Status:** ✅ Completed
**Current Version:** Final Submission (v2.0)

### Completed Tasks checklist
**Data Preparation**
- [x] Chose project topic: Cardiovascular Disease Prediction (Kaggle Dataset)
- [x] Downloaded raw `cardio_train.csv` (~70k records).
- [x] Inspected dataset shape, columns, and missing values.
- [x] Identified outliers (e.g., negative blood pressure, extreme BMI).
- [x] Cleaned dataset by dropping physiological impossibilities.
- [x] Converted `age` from days to years.
- [x] Saved cleaned dataset as `data/dataset_clean.csv`.
- [x] Created stratified 80/20 train-test split (`data/splits/`).

**Model Implementation (The Improvement Process)**
- [x] Implemented Naive Baseline (Majority Class) -> **50.87% Acc**.
- [x] Implemented Logistic Regression (Linear Baseline).
- [x] Implemented Linear SVM.
- [x] **Refinement 1**: Implemented **XGBoost** (Advanced Non-Linear Model).
- [x] **Refinement 2**: Feature Engineering (`BMI`, `MAP`, Interactions).

**Evaluation & Finalization**
- [x] Tuned model hyperparameters using GridSearchCV (F1-Score).
- [x] Compared Logistic Regression, SVM, XGBoost, and Baseline results.
- [x] Saved final metrics to `results/cv_results.json`.
- [x] Created figures (ROC Curves, Confusion Matrices, SHAP plots) for report.
- [x] Verified codebase runs reproducibly on teammates' machines.
- [x] Finalized README with setup instructions.
- [x] Prepared final report sections (Methodology, Results, Conclusion) in `docs/project_notes.md`.

## Key Findings: The Improvement Process

Aligned with the course requirement to "refine a solution", we demonstrated how **Algorithm Complexity** and **Domain Engineering** independently improve upon a naive baseline.

| Experiment Phase | Logistic Regression (Linear) | XGBoost (Non-Linear Tree) | Insight |
|------------------|------------------------------|---------------------------|---------|
| **Phase 1: Naive** | **50.87%** (Baseline) | N/A | Dataset is perfectly balanced; random chance is ~50%. |
| **Phase 2: Raw Features** | ~72.43% Accuracy | ~73.05% Accuracy | **Algorithmic Improvement**: Tree model naturally captures non-linearities (e.g., Weight/Height interaction). |
| **Phase 3: Engineered Features** | **72.55% Accuracy** | **73.11% Accuracy** | **Domain Improvement**: Explicitly creating `BMI` and `MAP` linearizes the problem, allowing simple models to catch up. |

**Conclusion**: Domain-driven feature engineering "linearized" the decision boundary, making computationally cheaper models (LogReg) nearly as effective as complex ones (XGBoost).

## Project Structure

```text
1513_Project/
  data/
    cardio_train.csv        # Raw Kaggle dataset
    dataset_clean.csv       # Preprocessed dataset (generated)
    splits/                 # Train/Test splits (80/20)
  docs/
    project_notes.md        # Detailed experiment log & discussion for report
    teammate_handoff.md     # Quick start guide
  results/
    cv_results.json         # Final 5-Fold Evaluation Metrics
    comparison_metrics.json # Model comparisons
    plots/                  # ROC, Confusion Matrices, SHAP plots
  src/
    preprocess.py           # Cleaning, Outlier Removal & Feature Engineering
    train_logreg.py         # Logistic Regression training script
    train_svm.py            # Linear SVM training script
    train_xgboost.py        # XGBoost training script
    evaluate_cv.py          # Main 5-Fold Cross-Validation Script
    evaluate_comparison.py  # Generates ROC/CM plots
    explainability.py       # Generates SHAP explainability plots
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data**:
   Cleans raw data and generates the engineered features (`BMI`, `MAP`, etc.).
   ```bash
   python src/preprocess.py
   ```

3. **Validation (5-Fold CV)**:
   Runs the rigorous evaluation suite and prints final accuracy/F1 scores.
   ```bash
   python src/evaluate_cv.py
   ```

4. **Generate Plots**:
   Creates the ROC curves and SHAP explanation plots found in `results/plots/`.
   ```bash
   python src/evaluate_comparison.py
   python src/explainability.py
   ```

