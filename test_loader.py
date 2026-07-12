from pathlib import Path
from src.data_loader import load_dataset

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

df = load_dataset(DATA_PATH)

print(df.head())
print(df.info())