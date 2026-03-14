import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "data/dataset_raw.csv"
CLEAN_DATA_PATH = "data/dataset_clean.csv"
SPLIT_DIR = "data/splits"

RANDOM_STATE = 42
TEST_SIZE = 0.2


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


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def save_data(df: pd.DataFrame, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_splits(X_train, X_test, y_train, y_test, split_dir: str) -> None:
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(split_path / "X_train.csv", index=False)
    X_test.to_csv(split_path / "X_test.csv", index=False)
    y_train.to_frame(name="target").to_csv(split_path / "y_train.csv", index=False)
    y_test.to_frame(name="target").to_csv(split_path / "y_test.csv", index=False)


if __name__ == "__main__":
    raw_df = load_data(RAW_DATA_PATH)
    inspect_data(raw_df, "Raw dataset")

    clean_df = clean_data(raw_df)
    inspect_data(clean_df, "Clean dataset")

    save_data(clean_df, CLEAN_DATA_PATH)
    print(f"\nSaved cleaned dataset to: {CLEAN_DATA_PATH}")

    X_train, X_test, y_train, y_test = split_data(clean_df)
    save_splits(X_train, X_test, y_train, y_test, SPLIT_DIR)

    print(f"\nSaved split files to: {SPLIT_DIR}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
