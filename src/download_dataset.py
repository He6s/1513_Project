from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Fetch official UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()

# Make sure y is a plain one-column dataframe
if isinstance(y, pd.Series):
    y = y.to_frame(name="num")

# Combine features and target into one dataframe
df = pd.concat([X, y], axis=1)

# Save original combined dataset
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "dataset_raw.csv", index=False)

print("Saved:", output_dir / "dataset_raw.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
