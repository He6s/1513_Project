# ECE1513 Final Project

Machine learning final project on heart disease prediction.

## Project overview

This project uses the UCI Heart Disease dataset to build and compare machine learning models for binary heart disease prediction.

Current progress
- dataset downloaded
- raw data inspected
- missing values handled
- target converted to binary
- train test split created
- naive baseline completed

## Dataset

Dataset used
- UCI Heart Disease dataset

Working data files
- `data/dataset_raw.csv`
- `data/dataset_clean.csv`

Target definition
- original target column was `num`
- new binary target column is `target`
- `0` means no heart disease
- `1` means heart disease present

Missing value handling
- rows with missing values were dropped
- missing values were originally found in `ca` and `thal`

## Project structure

```text
1513_Project/
  data/
    dataset_raw.csv
    dataset_clean.csv
    splits/
      X_train.csv
      X_test.csv
      y_train.csv
      y_test.csv
  docs/
    project_notes.md
    teammate_handoff.md
  results/
    baseline_results.json
    evaluate_test_results.json
  src/
    download_dataset.py
    preprocess.py
    baseline.py
    evaluate.py
  notebooks/
  README.md
  requirements.txt
```

## Environment setup

Create and activate the conda environment

```bash
conda create -n ece1513proj python=3.9 -y
conda activate ece1513proj
```

Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to run the current code

### 1. Download the dataset

```bash
python src/download_dataset.py
```

This saves
- `data/dataset_raw.csv`

### 2. Preprocess the dataset

```bash
python src/preprocess.py
```

This does the following
- loads the raw dataset
- drops rows with missing values
- converts the target to binary
- saves the cleaned dataset
- creates the train test split

This saves
- `data/dataset_clean.csv`
- `data/splits/X_train.csv`
- `data/splits/X_test.csv`
- `data/splits/y_train.csv`
- `data/splits/y_test.csv`

### 3. Run the naive baseline

```bash
python src/baseline.py
```

This saves
- `results/baseline_results.json`

## Current baseline result

Naive baseline using most frequent class.

Test set metrics
- Accuracy: 0.5333333333333333
- Precision: 0.0
- Recall: 0.0
- F1 score: 0.0

## Teammate instructions

For logistic regression and SVM, use these files
- `data/splits/X_train.csv`
- `data/splits/X_test.csv`
- `data/splits/y_train.csv`
- `data/splits/y_test.csv`

Everyone should use the same split files so model comparisons stay fair.

## Documentation

Working notes
- `docs/project_notes.md`

Teammate handoff
- `docs/teammate_handoff.md`

## Project task checklist

### Completed tasks
- Chose the project topic of heart disease prediction
- Selected the UCI Heart Disease dataset
- Downloaded the raw dataset
- Inspected dataset shape, columns, and missing values
- Identified missing values in `ca` and `thal`
- Cleaned the dataset by dropping rows with missing values
- Converted the original target column `num` into the binary target column `target`
- Saved the cleaned dataset as `data/dataset_clean.csv`
- Created a train test split
- Saved split files as
  - `data/splits/X_train.csv`
  - `data/splits/X_test.csv`
  - `data/splits/y_train.csv`
  - `data/splits/y_test.csv`
- Implemented a naive baseline using the most frequent class
- Evaluated the naive baseline and saved the results
- Added a shared evaluation helper for consistent model metrics
- Created project documentation notes
- Created a teammate handoff file
- Updated the README with setup and usage instructions

### To do
- Implement logistic regression training and evaluation
- Implement SVM training and evaluation
- Tune model hyperparameters if needed
- Compare logistic regression, SVM, and baseline results
- Save final result files for all models
- Create figures or tables for model comparison
- Make sure the full codebase is clean and well documented
- Check that the project runs correctly on teammates' computers
- Finalize the README if more files or steps are added
- Prepare the final report using the provided template
- Write the report sections based on the finished code and results
- Add dataset citation and other references to the report
- Complete teamwork attestation, AI usage, and consent sections
- Prepare the final presentation
