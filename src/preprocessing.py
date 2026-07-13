"""
preprocessing.py

Author: Arshiya

Description:
This module performs complete data preprocessing for the
Customer Churn Prediction project.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ----------------------------------------
# Directories
# ----------------------------------------

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------
# Main Function
# ----------------------------------------

def preprocess_data(df: pd.DataFrame):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    print("=" * 60)
    print("STARTING DATA PREPROCESSING")
    print("=" * 60)

    print("\nInitial Shape :", df.shape)

    # ----------------------------------------
    # Remove customerID
    # ----------------------------------------

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("customerID removed.")

    # ----------------------------------------
    # Convert TotalCharges
    # ----------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    missing_before = df["TotalCharges"].isna().sum()

    print(f"Missing TotalCharges : {missing_before}")

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # ----------------------------------------
    # Encode Target
    # ----------------------------------------

    df["Churn"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    # ----------------------------------------
    # Encode categorical variables
    # ----------------------------------------

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    # Save encoders

    joblib.dump(
        encoders,
        MODEL_DIR / "label_encoders.pkl"
    )

    print(f"Encoded Columns : {len(categorical_columns)}")

    # ----------------------------------------
    # Split Features & Target
    # ----------------------------------------

    X = df.drop(columns=["Churn"])

    y = df["Churn"]

    feature_names = X.columns.tolist()

    # ----------------------------------------
    # Train Test Split
    # ----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ----------------------------------------
    # Scale Features
    # ----------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        X_test_scaled,
        columns=feature_names,
        index=X_test.index
    )

    # ----------------------------------------
    # Save scaler
    # ----------------------------------------

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    # ----------------------------------------
    # Save processed data
    # ----------------------------------------

    train_df = X_train.copy()
    train_df["Churn"] = y_train.values

    test_df = X_test.copy()
    test_df["Churn"] = y_test.values

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    # ----------------------------------------
    # Summary
    # ----------------------------------------

    print("\nPreprocessing Completed Successfully")
    print("-" * 60)

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print(f"Features         : {X_train.shape[1]}")

    print(f"Train Shape      : {X_train.shape}")

    print(f"Test Shape       : {X_test.shape}")

    print("-" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )