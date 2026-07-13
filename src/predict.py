"""
predict.py

Prediction module for Customer Churn Prediction.
"""

import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_customer(customer_data: pd.DataFrame):

    feature_names = customer_data.columns

    customer_scaled = scaler.transform(customer_data)

    customer_scaled = pd.DataFrame(
        customer_scaled,
        columns=feature_names
    )

    prediction = model.predict(customer_scaled)[0]

    probability = model.predict_proba(customer_scaled)[0][1]

    return prediction, probability