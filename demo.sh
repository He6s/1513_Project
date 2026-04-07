#!/bin/bash
set -e

echo "Step 1. Preprocessing data and creating engineered-feature files"
python src/preprocess.py

echo
echo "Step 2. Running 10-seed hold-out comparison without feature engineering"
python src/evaluate_comparison_raw.py

echo
echo "Step 3. Running 10-seed hold-out comparison with engineered features"
python src/evaluate_comparison.py

echo
echo "Step 4. Running repeated cross-validation for the engineered-feature pipeline"
python src/evaluate_cv.py

echo
echo "Saved raw-feature comparison metrics"
cat results/comparison_metrics_raw.json

echo
echo "Saved engineered-feature comparison metrics"
cat results/comparison_metrics.json

echo
echo "Saved cross-validation metrics"
cat results/cv_results.json
