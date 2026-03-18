# ECE1513 Final Project

Machine learning final project on cardiovascular disease prediction.

## Project Overview

This project uses the Kaggle Cardiovascular Disease Dataset (~70,000 records) to build and compare machine learning models for binary heart disease prediction.
Previous experiments with the small UCI dataset (~300 records) showed performance saturation. The project was pivoted to this larger dataset to enable robust feature engineering and evaluation.

## Key Achievements
- **Data Pipeline**: End-to-end cleaning, outlier removal, and stratification.
- **Feature Engineering**: Integration of medical metrics (BMI, Pulse Pressure, MAP) and interaction terms.
- **Model Comparison**: Rigorous evaluation of Logistic Regression, Linear SVM, and XGBoost using 5-Fold Cross-Validation.
- **Explainability**: SHAP analysis to transparency in model decisions.

## Project Structure

```text
1513_Project/
  data/
    cardio_train.csv        # Original Kaggle dataset
    dataset_clean.csv       # Preprocessed dataset
    splits/                 # Train/Test splits (80/20)
  docs/
    project_notes.md        # Detailed experiment notes & final report data
    teammate_handoff.md     # Setup instructions
  results/
    baseline_results.json   # Dummy classifier results
    cv_results.json         # 5-Fold Cross-Validation metrics
    plots/                  # ROC curves, Confusion Matrices, SHAP plots
  src/
    preprocess.py           # Data cleaning & Feature Engineering
    train_*.py              # Individual model training scripts
    evaluate_cv.py          # 5-Fold Cross-Validation script
    evaluate_comparison.py  # Model comparison & plotting
    explainability.py       # SHAP analysis 
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   The `preprocess.py` script will clean data and generate splits.
   ```bash
   python src/preprocess.py
   ```

3. **Train & Evaluate**:
   Run the cross-validation script to see the final model performance.
   ```bash
   python src/evaluate_cv.py
   ```

4. **Generate Plots**:
   Create ROC curves, Confusion Matrices, and SHAP plots.
   ```bash
   python src/evaluate_comparison.py
   python src/explainability.py
   ```

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
