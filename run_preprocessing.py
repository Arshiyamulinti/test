from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

df = load_dataset(DATA_PATH)

X_train, X_test, y_train, y_test = preprocess_data(df)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)