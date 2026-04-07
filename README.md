# ECE1513 Project: Cardiovascular Disease Prediction
 
This repository contains our final ECE1513 course project on cardiovascular disease prediction using the Kaggle cardiovascular disease dataset.
 
## Repository contents
 
- `data/`
  - `cardio_train.csv`: raw Kaggle dataset
  - `dataset_clean.csv`: cleaned dataset after preprocessing
  - `splits/`: generated train and test split files
- `src/`
  - `preprocess.py`: data cleaning and feature engineering
  - `baseline.py`: naive majority-class baseline
  - `train_logreg.py`: logistic regression training
  - `train_svm.py`: linear SVM training
  - `train_xgboost.py`: XGBoost training
  - `evaluate.py`: shared evaluation helper
  - `evaluate_comparison.py`: 10-seed hold-out comparison
  - `evaluate_cv.py`: final repeated cross-validation evaluation
  - `explainability.py`: SHAP-based model interpretation
- `results/`
  - `comparison_metrics.json`: 10-seed averaged hold-out metrics
  - `cv_results.json`: final repeated cross-validation metrics
  - `plots/`: ROC, confusion matrix, and SHAP figures
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
 
### 10-seed averaged hold-out comparison
- Baseline: 50.88% ± 0.00%
- Logistic Regression: 72.11% ± 0.31%
- SVM linear: 72.07% ± 0.33%
- XGBoost: 72.94% ± 0.28%
 
### Final repeated cross-validation results
- Logistic Regression: accuracy 0.7246, F1 0.7198
- SVM linear: accuracy 0.7242, F1 0.7204
- XGBoost: accuracy 0.7313, F1 0.7246
 
XGBoost achieved the best overall performance, but the margin over the linear models was small.
 
## Setup
 
We recommend Python 3.11.
 
### Conda
 
    conda create -n ece1513proj python=3.11 -y
    conda activate ece1513proj
    pip install -r requirements.txt
 
## Usage
 
### Step 1. Preprocess the dataset
 
    python src/preprocess.py
 
### Step 2. Run the 10-seed hold-out comparison
 
    python src/evaluate_comparison.py
 
### Step 3. Run the final repeated cross-validation evaluation
 
    python src/evaluate_cv.py
 
### Step 4. Generate explainability plots
 
    python src/explainability.py
 
## Demo
 
To run a simple end-to-end demo:
 
    bash demo.sh
 
This script:
- preprocesses the data
- runs the 10-seed hold-out comparison
- runs the final repeated cross-validation evaluation
- prints the saved metrics files

