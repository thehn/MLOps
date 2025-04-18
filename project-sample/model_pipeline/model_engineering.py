# model_training.py: trains new models using MLflow and logs the model.
# This script includes loading data from feature_store in SQLite database
# and training a model using different algorithms.
# Input: SQLite database file containing features.
# Output: MLflow model logged to the tracking server.


import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import sqlite3
import argparse


def load_data(database_file, table_name):
    """
    Load data from SQLite database.

    Args:
        database_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to load data from.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    conn = sqlite3.connect(database_file)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and splitting into features and target variable.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.

    Returns:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
    """
    # Handle missing values
    df = df.dropna()

    # Convert categorical variables to numeric
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes

    # Split into features and target variable
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    return X, y


def feature_selection(X, y, k=10):
    """
    Perform feature selection using SelectKBest.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
        k (int): Number of top features to select.

    Returns:
        X_selected (pd.DataFrame): DataFrame with selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    return X_selected


def train_model(X, y, model_type="RandomForest"):
    """
    Train a model using the specified model type.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
        model_type (str): Type of model to train. Options: "RandomForest", "LogisticRegression", "SVC".

    Returns:
        model: Trained model.
    """
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=42)
    elif model_type == "SVC":
        model = SVC(random_state=42)
    else:
        raise ValueError(
            "Invalid model type. Choose from 'RandomForest', 'LogisticRegression', or 'SVC'."
        )

    model.fit(X, y)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, and F1 score.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features DataFrame.
        y_test (pd.Series): Test target variable Series.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    return metrics


def log_model(model, model_name, metrics):
    """
    Log the model and its metrics to MLflow.

    Args:
        model: Trained model.
        model_name (str): Name of the model.
        metrics (dict): Dictionary containing evaluation metrics.
    """
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_params({"model_name": model_name})
        mlflow.log_metrics(metrics)
        print(f"Model {model_name} logged to MLflow.")


# Example usage
if __name__ == "__main__":
    # Load data from SQLite database
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Train and log a machine learning model."
    )
    parser.add_argument(
        "--database-file",
        type=str,
        required=True,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        required=True,
        help="Name of the table to load data from.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "LogisticRegression", "SVC"],
        help="Type of model to train.",
    )
    args = parser.parse_args()

    database_file = args.database_file
    table_name = args.table_name
    model_type = args.model_type
    df = load_data(database_file, table_name)

    # Preprocess data
    X, y = preprocess_data(df)

    # Feature selection
    X_selected = feature_selection(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train, model_type)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: {metrics}")

    # Log model to MLflow
    log_model(model, model_type, metrics)
