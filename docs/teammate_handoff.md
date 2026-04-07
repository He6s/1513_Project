# Teammate Handoff

## Final branch
Use the Kaggle project branch for the final submission version.

## Required files
- `data/cardio_train.csv`
- `requirements.txt`

## Quick setup
```bash
conda create -n ece1513proj python=3.11 -y
conda activate ece1513proj
pip install -r requirements.txt
Quick run
python src/preprocess.py
python src/evaluate_comparison.py
python src/evaluate_cv.py
python src/explainability.py
Output files
results/comparison_metrics.json
results/cv_results.json
results/plots/
Final model summary

XGBoost is the best overall model, with only a small margin over logistic regression and linear SVM after feature engineering.
