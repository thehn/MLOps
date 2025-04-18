# model_validation.py
import pandas as pd
import mlflow
import argparse
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Validate the model using test data before deployment
def validate_model(model_uri: str, test_data_path: str):
    """
    Validate the model using test data before deployment.

    Args:
        model_uri (str): URI of the model in MLflow.
        test_data_path (str): Path to the test data CSV file.

    Returns:
        None
    """

    # Load test data
    test = pd.read_csv(test_data_path)
    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    # Load the trained model from MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # Make predictions
    predictions = model.predict(X_test)
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log evaluation metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    # Print evaluation results
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1 Score: {f1}")

    # Save the validation report
    validation_report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    report_df = pd.DataFrame([validation_report])
    report_df.to_csv("validation_report.csv", index=False)
    print("Validation report saved as validation_report.csv")

    # Perform additional checks (e.g., fairness metrics or robustness tests)
    print("Model validation completed.")


# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate a trained ML model.")
    parser.add_argument(
        "--model-uri", type=str, required=True, help="URI of the model in MLflow."
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to the test data CSV file.",
    )
    args = parser.parse_args()

    model_uri = args.model_uri
    test_data_path = args.test_data_path

    validate_model(model_uri, test_data_path)
