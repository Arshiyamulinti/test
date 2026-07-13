from pathlib import Path

from src.data_loader import load_dataset

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

df = load_dataset(DATA_PATH)

print("="*60)
print("Dataset Shape")
print("="*60)
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nTarget Variable")
print(df["Churn"].value_counts())

print("\nSummary Statistics")
print(df.describe(include="all"))