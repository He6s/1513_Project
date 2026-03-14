import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("Shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    path = "data/dataset.csv"
    df = load_data(path)
    inspect_data(df)
