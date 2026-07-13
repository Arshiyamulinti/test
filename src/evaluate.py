"""
evaluate.py

Model evaluation utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, probabilities),
    }

    return metrics


def generate_reports(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Classification Report
    report = classification_report(y_test, predictions)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Metrics
    with open("reports/metrics.txt", "w") as f:

        f.write(f"Accuracy : {accuracy_score(y_test,predictions):.4f}\n")
        f.write(f"Precision : {precision_score(y_test,predictions):.4f}\n")
        f.write(f"Recall : {recall_score(y_test,predictions):.4f}\n")
        f.write(f"F1 Score : {f1_score(y_test,predictions):.4f}\n")
        f.write(f"ROC AUC : {roc_auc_score(y_test,probabilities):.4f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig(
        "reports/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ROC Curve

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.savefig(
        "reports/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nEvaluation reports generated successfully!")