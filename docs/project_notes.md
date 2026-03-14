# ECE1513 Final Project Notes

## Project topic
Heart disease prediction using machine learning.

## Dataset
Current dataset file: `data/dataset_raw.csv`

### Initial dataset inspection
- Number of rows: 303
- Number of columns: 14
- Columns:
  - age
  - sex
  - cp
  - trestbps
  - chol
  - fbs
  - restecg
  - thalach
  - exang
  - oldpeak
  - slope
  - ca
  - thal
  - num

### Observed missing values
- `ca`: 4 missing values
- `thal`: 2 missing values

### Current target column
- `num`

### Planned prediction task
Convert `num` into a binary target:
- 0 means no heart disease
- 1 means heart disease present

### Planned preprocessing steps
- Load raw dataset
- Inspect missing values
- Drop rows with missing values in `ca` and `thal`
- Convert target `num` to binary
- Save cleaned dataset
- Later split into train and test sets

## Preprocessing progress
Completed:
- Loaded raw dataset successfully
- Confirmed shape and columns
- Confirmed missing values in `ca` and `thal`
- Implemented row removal for missing values
- Converted `num` into binary target column named `target`
- Saved cleaned dataset to `data/dataset_clean.csv`

## Current code files
- `src/download_dataset.py`
- `src/preprocess.py`

## Current data files
- `data/dataset_raw.csv`
- `data/dataset_clean.csv`
