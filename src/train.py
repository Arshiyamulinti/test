"""
train.py

Train multiple machine learning models,
compare their performance,
save the best model,
and generate evaluation reports.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.model_factory import get_models
from src.evaluate import (
    evaluate_model,
    generate_reports,
)


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

DATASET_PATH = Path("data/raw/Telco-Customer-Churn.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# Training Function
# ---------------------------------------------------

def train_models():

    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    df = load_dataset(DATASET_PATH)

    # ---------------------------------------------
    # Preprocess Dataset
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # ---------------------------------------------
    # Load Models
    # ---------------------------------------------

    models = get_models()

    results = []

    best_model = None
    best_model_name = None
    best_accuracy = 0

    print("\nTraining Models...\n")

    # ---------------------------------------------
    # Train Every Model
    # ---------------------------------------------

    for model_name, model in models.items():

        print("-" * 50)
        print(f"Training : {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "ROC AUC": metrics["ROC AUC"]
        })

        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1 Score']:.4f}")
        print(f"ROC AUC  : {metrics['ROC AUC']:.4f}")

        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_model = model
            best_model_name = model_name

    # ---------------------------------------------
    # Save Comparison Table
    # ---------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    comparison_file = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(
        comparison_file,
        index=False
    )

    # ---------------------------------------------
    # Save Best Model
    # ---------------------------------------------

    model_file = MODEL_DIR / "model.pkl"

    joblib.dump(
        best_model,
        model_file
    )

    # ---------------------------------------------
    # Generate Reports
    # ---------------------------------------------

    generate_reports(
        best_model,
        X_test,
        y_test
    )

    # ---------------------------------------------
    # Final Summary
    # ---------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df)

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model Name : {best_model_name}")
    print(f"Accuracy   : {best_accuracy:.4f}")

    print("\nSaved Files")

    print(f"✔ Model      : {model_file}")
    print(f"✔ Comparison : {comparison_file}")
    print("✔ Classification Report")
    print("✔ Metrics Report")
    print("✔ Confusion Matrix")
    print("✔ ROC Curve")

    print("\nProject Training Completed Successfully!")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    train_models()