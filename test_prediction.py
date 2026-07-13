import pandas as pd

from src.predict import predict_customer

sample_customer = pd.DataFrame(
    [{
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 24,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 1,
        "OnlineBackup": 1,
        "DeviceProtection": 1,
        "TechSupport": 1,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract": 1,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0
    }]
)

prediction, probability = predict_customer(sample_customer)

print("=" * 60)

if prediction == 1:
    print("Prediction : Customer Will Churn")
else:
    print("Prediction : Customer Will Stay")

print(f"Probability : {probability:.2%}")
