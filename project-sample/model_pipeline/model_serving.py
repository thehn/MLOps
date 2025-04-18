# Serve a trained model using MLflow

import mlflow
import mlflow.pyfunc
import pandas as pd


# Find the best model from MLflow
def find_best_model():
    """
    Find the best model from MLflow.

    Returns:
        str: URI of the best model.
    """
    # Get the list of all runs
    runs = mlflow.search_runs(order_by=["metrics.test_accuracy desc"])
    best_run = runs.iloc[0]
    best_model_uri = f"runs:/{best_run.run_id}/model"
    return best_model_uri


def load_model(model_uri: str):
    """
    Load a model from MLflow.

    Args:
        model_uri (str): URI of the model in MLflow.

    Returns:
        model: Loaded model.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def serve_model(model_uri: str, input_data: pd.DataFrame):
    """
    Serve a model using MLflow.

    Args:
        model_uri (str): URI of the model in MLflow.
        input_data (pd.DataFrame): Input data for prediction.

    Returns:
        pd.DataFrame: Predictions made by the model.
    """
    # Load the model
    model = load_model(model_uri)

    # Make predictions
    predictions = model.predict(input_data)
    return predictions


# serve model as a REST API
def serve_model_rest_api(model_uri: str):
    """
    Serve a model as a REST API using Flask.

    Args:
        model_uri (str): URI of the model in MLflow.
        input_data (pd.DataFrame): Input data for prediction.

    Returns:
        str: JSON response with predictions.
    """
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        input_df = pd.DataFrame(data)
        predictions = serve_model(model_uri, input_df)
        return jsonify(predictions.tolist())

    return app


if __name__ == "__main__":
    # Example usage
    model_uri = find_best_model()

    # Load the model
    model = load_model(model_uri)
    print(f"Model loaded from {model_uri}")

    input_data = pd.read_csv("input_data.csv")  # Replace with your input data file
    predictions = serve_model(model_uri, input_data)
    print(predictions)

    # Serve the model as a REST API
    app = serve_model_rest_api(model_uri)
