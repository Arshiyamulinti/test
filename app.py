import streamlit as st
import pandas as pd

from src.predict import predict_customer

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")

st.write("Predict whether a telecom customer is likely to churn.")

st.divider()

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

if st.button("Predict"):

    customer = pd.DataFrame([{

        "gender": 1 if gender == "Male" else 0,

        "SeniorCitizen": 1 if senior == "Yes" else 0,

        "Partner": 1 if partner == "Yes" else 0,

        "Dependents": 1 if dependents == "Yes" else 0,

        "tenure": tenure,

        "PhoneService": 1 if phone_service == "Yes" else 0,

        "MultipleLines": {
            "No": 0,
            "Yes": 1,
            "No phone service": 2
        }[multiple_lines],

        "InternetService": {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }[internet],

        "OnlineSecurity": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[online_security],

        "OnlineBackup": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[online_backup],

        "DeviceProtection": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[device_protection],

        "TechSupport": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[tech_support],

        "StreamingTV": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[streaming_tv],

        "StreamingMovies": {
            "No": 0,
            "Yes": 1,
            "No internet service": 2
        }[streaming_movies],

        "Contract": {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }[contract],

        "PaperlessBilling": 1 if paperless == "Yes" else 0,

        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }[payment],

        "MonthlyCharges": monthly,

        "TotalCharges": total

    }])

    prediction, probability = predict_customer(customer)

    st.divider()

    if prediction == 1:
        st.error("❌ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )