import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# Constants
RAW_DATA_PATH = "data/dataset_raw.csv"
CLEAN_DATA_PATH = "data/dataset_clean.csv"
SPLIT_DIR = Path("data/splits")

# Kaggle dataset often uses semicolon delimiter
DELIMITER = ";" 

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset with correct delimiter."""
    # The file is expected to be 'cardio_train.csv' renamed to 'dataset_raw.csv'
    # It typically uses ';' as separator
    try:
        # Try finding the file even if path is slightly off or relative
        p = Path(filepath)
        if not p.exists():
             # Fallback to check if cardio_train exists directly via user hint
             if Path("data/cardio_train.csv").exists():
                 p = Path("data/cardio_train.csv")
             else:
                 raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading from: {p}")
        df = pd.read_csv(p, sep=DELIMITER)
        
        # Check if delimiter was wrong (collapsed into 1 column)
        if df.shape[1] == 1:
             print("Warning: Semicolon delimiter failed (1 column found). Trying comma.")
             df = pd.read_csv(p, sep=",")

        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning specific to the Cardiovascular Disease dataset.
    """
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Drop 'id' column as it's not predictive
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 2. Convert Age from days to years (rounding down)
    if "age" in df.columns:
        df["age"] = (df["age"] / 365.25).round().astype(int)

    # 3. Handle Duplicates
    df = df.drop_duplicates()
    
    # 4. Handle Outliers / Impossible values (Crucial for this dataset!)
    
    # Blood Pressure: 
    # ap_hi (Systolic) should be higher than ap_lo (Diastolic)
    # Reasonable range: ap_hi [60, 240], ap_lo [40, 160]
    # Filter out negative pressures and unphysiological values
    mask_bp = (df['ap_hi'] >= 60) & (df['ap_hi'] <= 240) & \
              (df['ap_lo'] >= 40) & (df['ap_lo'] <= 160) & \
              (df['ap_hi'] > df['ap_lo'])
    
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df = df[mask_bp]

    # Height and Weight:
    # Filter out extreme outliers (e.g. height < 100cm or weight < 30kg for adults)
    mask_body = (df['height'] >= 100) & (df['height'] <= 250) & \
                (df['weight'] >= 30) & (df['weight'] <= 200)
    
    if 'height' in df.columns and 'weight' in df.columns:
        df = df[mask_body]

    print(f"Data cleaning complete.")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Removed {initial_rows - len(df)} rows due to outliers/duplicates.")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features to improve model performance.
    """
    df = df.copy()
    initial_rows = len(df)
    print("\nAdding new features...")

    # 1. Body Mass Index (BMI)
    # BMI = weight (kg) / height (m)^2
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # 2. Pulse Pressure (PP)
    # PP = Systolic - Diastolic. High PP is a risk factor.
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # 3. Mean Arterial Pressure (MAP)
    # MAP = (SBP + 2*DBP) / 3
    # Approximates the average pressure in a patient's arteries during one cardiac cycle.
    df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

    # 4. Interaction Terms
    # Age is a key risk factor, combining it with cholesterol amplifies risk.
    df['age_chol_interaction'] = df['age'] * df['cholesterol']
    
    # 5. Filter BMI Outliers (Stricter filtering)
    # Filter out extreme BMI values which are likely data errors
    # 10 < BMI < 60 is a very broad physiological range
    mask_bmi = (df['bmi'] > 10) & (df['bmi'] < 60)
    df = df[mask_bmi]

    print(f"Feature engineering complete.")
    print(f"Rows after BMI filtering: {len(df)}")
    print(f"Removed {initial_rows - len(df)} rows based on BMI.")

    return df


def save_splits(df: pd.DataFrame):
    """Split data and save to CSVs."""
    target_col = "cardio"
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    X = df.drop(columns=[target_col])
    # Rename 'cardio' to 'target' for consistency with training scripts
    y = df[[target_col]].rename(columns={target_col: "target"}) 

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving splits...")
    X_train.to_csv(SPLIT_DIR / "X_train.csv", index=False)
    X_test.to_csv(SPLIT_DIR / "X_test.csv", index=False)
    y_train.to_csv(SPLIT_DIR / "y_train.csv", index=False)
    y_test.to_csv(SPLIT_DIR / "y_test.csv", index=False)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Files saved to {SPLIT_DIR}")


if __name__ == "__main__":
    # Pipeline
    df_raw = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)
    
    # Feature Engineering
    df_features = add_features(df_clean)
    
    # Save full clean dataset (optional)
    clean_path = Path(CLEAN_DATA_PATH)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(clean_path, index=False)
    
    # Split
    save_splits(df_features)
