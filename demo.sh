#!/bin/bash
set -e

echo "Step 1. Preprocessing data"
python src/preprocess.py

echo
echo "Step 2. Running 10-seed hold-out comparison"
python src/evaluate_comparison.py

echo
echo "Step 3. Running repeated cross-validation"
python src/evaluate_cv.py

echo
echo "Saved comparison metrics"
cat results/comparison_metrics.json

echo
echo "Saved cross-validation metrics"
cat results/cv_results.json
