# Teammate Handoff

## Current project status
The dataset has already been downloaded, cleaned, converted to binary classification, and split into train and test sets.

## Dataset being used
Official UCI Heart Disease dataset

Main working files:
- Raw dataset: `data/dataset_raw.csv`
- Clean dataset: `data/dataset_clean.csv`

## Prediction target
The original target column was `num`.

It was converted into a binary target column called `target`:
- `0` means no heart disease
- `1` means heart disease present

## Missing value handling
Rows with missing values were dropped.

Missing values were originally found in:
- `ca`
- `thal`

## Train test split
The split is already created and saved.

Settings used:
- test size = 0.2
- random state = 42
- stratified split on target

Saved files:
- `data/splits/X_train.csv`
- `data/splits/X_test.csv`
- `data/splits/y_train.csv`
- `data/splits/y_test.csv`

## Baseline
A naive baseline has already been created using the most frequent class.

Code:
- `src/baseline.py`

Saved results:
- `results/baseline_results.json`

Baseline test metrics:
- Accuracy: 0.5333333333333333
- Precision: 0.0
- Recall: 0.0
- F1 score: 0.0

## Files teammates should use
For logistic regression and SVM, use these split files:
- `data/splits/X_train.csv`
- `data/splits/X_test.csv`
- `data/splits/y_train.csv`
- `data/splits/y_test.csv`

## Important note
Everyone should use the same split files so all models are compared fairly.
