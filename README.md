# ECE1513 Project: Cardiovascular Disease Prediction

This repository contains our final ECE1513 course project on cardiovascular disease prediction using the Kaggle cardiovascular disease dataset.

## Repository contents

- `data/`
  - `cardio_train.csv`: raw Kaggle dataset
  - `dataset_clean.csv`: cleaned dataset with engineered features
  - `splits/`: generated train and test split files for the engineered-feature pipeline
- `src/`
  - `preprocess.py`: data cleaning and feature engineering
  - `baseline.py`: naive majority-class baseline
  - `train_logreg.py`: logistic regression training
  - `train_svm.py`: linear SVM training
  - `train_xgboost.py`: XGBoost training
  - `evaluate.py`: shared evaluation helper
  - `evaluate_comparison.py`: 10-seed hold-out comparison with engineered features
  - `evaluate_comparison_raw.py`: 10-seed hold-out comparison without feature engineering
  - `evaluate_cv.py`: final repeated cross-validation evaluation for the engineered-feature pipeline
  - `explainability.py`: SHAP-based model interpretation
- `results/`
  - `comparison_metrics_raw.json`: 10-seed averaged hold-out metrics without feature engineering
  - `comparison_metrics.json`: 10-seed averaged hold-out metrics with feature engineering
  - `cv_results.json`: final repeated cross-validation metrics for the engineered-feature pipeline
  - `plots/`: representative ROC, confusion matrix, and SHAP figures used for the report
- `docs/`
  - `project_notes.md`: short project summary
  - `teammate_handoff.md`: quick run guide
- `requirements.txt`: required Python packages
- `demo.sh`: demo script for sample input-output

## Method summary

We treated cardiovascular disease prediction as a binary classification problem. The pipeline includes:
- data cleaning
- feature engineering
- logistic regression
- linear SVM
- XGBoost
- evaluation over 10 random seeds
- SHAP-based explainability

Engineered features include:
- BMI
- pulse pressure
- mean arterial pressure
- age-cholesterol interaction

## Final results

### 10-seed averaged hold-out comparison without feature engineering
- Baseline: 50.87% ± 0.00%
- Logistic Regression: 72.21% ± 0.19%
- SVM linear: 72.17% ± 0.21%
- XGBoost: 72.95% ± 0.22%

### 10-seed averaged hold-out comparison with feature engineering
- Baseline: 50.88% ± 0.00%
- Logistic Regression: 72.11% ± 0.31%
- SVM linear: 72.07% ± 0.33%
- XGBoost: 72.94% ± 0.28%

### Final repeated cross-validation results for the engineered-feature pipeline
- Logistic Regression: accuracy 0.7246, F1 0.7198
- SVM linear: accuracy 0.7242, F1 0.7204
- XGBoost: accuracy 0.7313, F1 0.7246

## Takeaway

All learned models clearly outperform the naive baseline. Under 10-seed evaluation, feature engineering has only a marginal effect on hold-out performance. It slightly changes the linear-model F1 scores while leaving XGBoost almost unchanged overall.

## Setup

We recommend Python 3.11.

### Conda
```bash
conda create -n ece1513proj python=3.11 -y
conda activate ece1513proj
pip install -r requirements.txt
```

## Usage

### Step 1. Preprocess the dataset and create the engineered-feature files
```bash
python src/preprocess.py
```

### Step 2. Run the 10-seed hold-out comparison without feature engineering
```bash
python src/evaluate_comparison_raw.py
```

### Step 3. Run the 10-seed hold-out comparison with engineered features
```bash
python src/evaluate_comparison.py
```

### Step 4. Run the final repeated cross-validation evaluation for the engineered-feature pipeline
```bash
python src/evaluate_cv.py
```

### Step 5. Generate explainability plots
```bash
python src/explainability.py
```

## Demo

To run a simple end-to-end demo:
```bash
bash demo.sh
```

This script:
- preprocesses the data
- runs the 10-seed engineered-feature hold-out comparison
- runs the final repeated cross-validation evaluation
- prints the saved metrics files

## Notes

- This repository reflects the final Kaggle-based version of the project.
- The final submission branch is `feature/kaggle-dataset-xgb`.
- The plots in `results/plots/` are kept as representative figures for the report.
