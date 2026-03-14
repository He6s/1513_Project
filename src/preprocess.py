import pandas as pd
from pathlib import Path


RAW_DATA_PATH = "data/dataset_raw.csv"
CLEAN_DATA_PATH = "data/dataset_clean.csv"


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def inspect_data(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print("Shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing values
    df = df.dropna()

    # Convert target to binary
    df["target"] = (df["num"] > 0).astype(int)

    # Drop original multiclass target
    df = df.drop(columns=["num"])

    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    raw_df = load_data(RAW_DATA_PATH)
    inspect_data(raw_df, "Raw dataset")

    clean_df = clean_data(raw_df)
    inspect_data(clean_df, "Clean dataset")

    save_data(clean_df, CLEAN_DATA_PATH)
    print(f"\nSaved cleaned dataset to: {CLEAN_DATA_PATH}")
