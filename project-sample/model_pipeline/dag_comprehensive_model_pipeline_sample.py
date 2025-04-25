# This Airflow DAG represents a complete MLOps pipeline that integrates MLflow for experiment tracking and model registry. Here's a breakdown of what it does:

# Data Extraction: Generates synthetic customer churn data
# Data Preprocessing: Cleans and normalizes the data
# Feature Engineering: Creates additional features and encodes categorical variables
# Train/Test Split: Divides data for model training and evaluation
# Model Training: Trains a RandomForest classifier and logs it to MLflow
# Model Evaluation: Calculates metrics and logs them to MLflow
# Model Promotion: Transitions the model to Production stage in MLflow if it meets quality thresholds
# Model Serving Setup: Prepares a Flask API for model deployment
# Monitoring Setup: Creates a script for tracking model performance

# To use this DAG:

# Make sure you have Airflow and MLflow servers running
# Update MLFLOW_TRACKING_URI with your MLflow server address
# Place this file in your Airflow DAGs folder

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import joblib

# Define default arguments for the DAG
default_args = {
    "owner": "mlops_user",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email": ["mlops_user@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    "mlops_end_to_end_pipeline",
    default_args=default_args,
    description="End-to-end MLOps pipeline with MLflow integration",
    schedule_interval=timedelta(days=1),
    tags=["mlops", "training", "deployment"],
)

# Define MLflow tracking URI - update this with your MLflow server details
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "customer_churn_prediction"
MODEL_NAME = "churn_model"


# Define functions for each stage of the MLOps pipeline
def data_extraction(**kwargs):
    """Extract data from source."""
    # In a real scenario, you would connect to a database or API
    # For this example, we'll generate synthetic data
    print("Extracting data...")

    np.random.seed(42)
    n_samples = 1000

    # Generate features
    age = np.random.normal(45, 15, n_samples)
    tenure = np.random.poisson(15, n_samples)
    monthly_charges = np.random.normal(65, 30, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)

    # Binary features
    contract_type = np.random.choice(
        [0, 1, 2], n_samples
    )  # Month-to-month, One year, Two year
    payment_method = np.random.choice([0, 1, 2, 3], n_samples)
    tech_support = np.random.choice([0, 1], n_samples)
    online_backup = np.random.choice([0, 1], n_samples)

    # Generate target (churn)
    logits = (
        -2
        + 0.04 * (65 - age)
        - 0.2 * tenure
        + 0.01 * monthly_charges
        - 0.8 * contract_type
        - 0.4 * tech_support
    )
    prob_churn = 1 / (1 + np.exp(-logits))
    churn = (np.random.random(n_samples) < prob_churn).astype(int)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "tech_support": tech_support,
            "online_backup": online_backup,
            "churn": churn,
        }
    )

    # Save data to a temporary location
    data_path = "/tmp/churn_data.csv"
    data.to_csv(data_path, index=False)
    print(f"Data extracted and saved to {data_path}")

    # Push the data path to XCom for the next task
    kwargs["ti"].xcom_push(key="data_path", value=data_path)
    return data_path


def data_preprocessing(**kwargs):
    """Clean and preprocess the data."""
    print("Preprocessing data...")

    # Pull data path from XCom
    ti = kwargs["ti"]
    data_path = ti.xcom_pull(task_ids="data_extraction", key="data_path")

    # Load data
    data = pd.read_csv(data_path)

    # Example preprocessing steps
    # Handle outliers in total charges
    q1 = data["total_charges"].quantile(0.25)
    q3 = data["total_charges"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data["total_charges"] = np.where(
        (data["total_charges"] < lower_bound) | (data["total_charges"] > upper_bound),
        data["total_charges"].median(),
        data["total_charges"],
    )

    # Normalize numerical features
    for col in ["age", "tenure", "monthly_charges", "total_charges"]:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # Save preprocessed data
    processed_data_path = "/tmp/churn_data_processed.csv"
    data.to_csv(processed_data_path, index=False)
    print(f"Data preprocessed and saved to {processed_data_path}")

    # Push the processed data path to XCom
    kwargs["ti"].xcom_push(key="processed_data_path", value=processed_data_path)
    return processed_data_path


def feature_engineering(**kwargs):
    """Create and select features."""
    print("Engineering features...")

    # Pull data path from XCom
    ti = kwargs["ti"]
    data_path = ti.xcom_pull(task_ids="data_preprocessing", key="processed_data_path")

    # Load data
    data = pd.read_csv(data_path)

    # Feature engineering examples
    # Create interaction terms
    data["tenure_contract"] = data["tenure"] * data["contract_type"]
    data["charges_ratio"] = data["monthly_charges"] / (data["total_charges"] + 1)

    # One-hot encoding for categorical variables
    data = pd.get_dummies(
        data, columns=["contract_type", "payment_method"], drop_first=True
    )

    # Save the data with engineered features
    featured_data_path = "/tmp/churn_data_featured.csv"
    data.to_csv(featured_data_path, index=False)
    print(f"Feature engineering completed and saved to {featured_data_path}")

    # Push the featured data path to XCom
    kwargs["ti"].xcom_push(key="featured_data_path", value=featured_data_path)
    return featured_data_path


def train_test_split_task(**kwargs):
    """Split data into training and testing sets."""
    print("Splitting data into train and test sets...")

    # Pull data path from XCom
    ti = kwargs["ti"]
    data_path = ti.xcom_pull(task_ids="feature_engineering", key="featured_data_path")

    # Load data
    data = pd.read_csv(data_path)

    # Split features and target
    X = data.drop("churn", axis=1)
    y = data["churn"]

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save splits
    train_path = "/tmp/train_data.pkl"
    test_path = "/tmp/test_data.pkl"

    joblib.dump((X_train, y_train), train_path)
    joblib.dump((X_test, y_test), test_path)

    print(f"Data split and saved to {train_path} and {test_path}")

    # Push paths to XCom
    kwargs["ti"].xcom_push(key="train_path", value=train_path)
    kwargs["ti"].xcom_push(key="test_path", value=test_path)
    return {"train_path": train_path, "test_path": test_path}


def model_training(**kwargs):
    """Train and log the model using MLflow."""
    print("Training model...")

    # Pull data paths from XCom
    ti = kwargs["ti"]
    train_path = ti.xcom_pull(task_ids="train_test_split_task", key="train_path")

    # Load training data
    X_train, y_train = joblib.load(train_path)

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run(run_name="rf_model_training") as run:
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Train model
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        # Log feature importance
        feature_importance = pd.DataFrame(
            rf_model.feature_importances_, index=X_train.columns, columns=["importance"]
        ).sort_values("importance", ascending=False)

        # Log model with signature
        signature = infer_signature(X_train, rf_model.predict(X_train))
        mlflow.sklearn.log_model(
            rf_model,
            "random_forest_model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # Save run_id for later tasks
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/random_forest_model"

        # Log artifact paths
        model_path = f"/tmp/rf_model_{run_id}.pkl"
        joblib.dump(rf_model, model_path)

        print(f"Model trained and logged to MLflow with run_id: {run_id}")
        print(f"Model URI: {model_uri}")

        # Push to XCom
        kwargs["ti"].xcom_push(key="run_id", value=run_id)
        kwargs["ti"].xcom_push(key="model_uri", value=model_uri)
        kwargs["ti"].xcom_push(key="model_path", value=model_path)

    return {"run_id": run_id, "model_uri": model_uri, "model_path": model_path}


def model_evaluation(**kwargs):
    """Evaluate model performance and log metrics."""
    print("Evaluating model...")

    # Pull data from XCom
    ti = kwargs["ti"]
    test_path = ti.xcom_pull(task_ids="train_test_split_task", key="test_path")
    run_id = ti.xcom_pull(task_ids="model_training", key="run_id")
    model_path = ti.xcom_pull(task_ids="model_training", key="model_path")

    # Load test data and model
    X_test, y_test = joblib.load(test_path)
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Log metrics to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

        # Generate and log confusion matrix as a figure
        cm_path = f"/tmp/confusion_matrix_{run_id}.png"
        # In a real scenario, you would generate and save the confusion matrix image
        # For this example, we'll just simulate it
        with open(cm_path, "w") as f:
            f.write("Placeholder for confusion matrix image")

        mlflow.log_artifact(cm_path, "evaluation")

    print(f"Model evaluation complete. Metrics: {metrics}")

    # Push evaluation results to XCom
    kwargs["ti"].xcom_push(key="evaluation_metrics", value=metrics)

    return metrics


def model_promotion(**kwargs):
    """Promote model to production if it meets criteria."""
    print("Evaluating model for promotion...")

    # Pull data from XCom
    ti = kwargs["ti"]
    run_id = ti.xcom_pull(task_ids="model_training", key="run_id")
    metrics = ti.xcom_pull(task_ids="model_evaluation", key="evaluation_metrics")
    model_uri = ti.xcom_pull(task_ids="model_training", key="model_uri")

    # Define promotion criteria
    promotion_threshold = 0.75  # Example threshold for f1_score

    # Check if model meets criteria
    if metrics["f1_score"] >= promotion_threshold:
        print(
            f"Model meets promotion criteria (f1_score: {metrics['f1_score']} >= {promotion_threshold})"
        )

        # Set up MLflow client
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Get the latest version
        latest_versions = client.get_latest_versions(MODEL_NAME)
        latest_version = max([int(v.version) for v in latest_versions])

        # Transition the model to Production
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest_version, stage="Production"
        )

        print(f"Model version {latest_version} transitioned to Production stage")
        production_model_uri = f"models:/{MODEL_NAME}/Production"

        # Push production model URI to XCom
        kwargs["ti"].xcom_push(key="production_model_uri", value=production_model_uri)

        return {
            "promoted": True,
            "version": latest_version,
            "uri": production_model_uri,
        }
    else:
        print(
            f"Model does not meet promotion criteria (f1_score: {metrics['f1_score']} < {promotion_threshold})"
        )
        return {"promoted": False}


def model_serving_setup(**kwargs):
    """Set up model serving infrastructure."""
    print("Setting up model serving...")

    # Pull data from XCom
    ti = kwargs["ti"]
    promotion_result = ti.xcom_pull(task_ids="model_promotion")

    if promotion_result and promotion_result.get("promoted", False):
        production_model_uri = promotion_result["uri"]

        # In a real scenario, this might:
        # 1. Deploy to a Flask/FastAPI service
        # 2. Update a Kubernetes deployment
        # 3. Push to a model serving platform like SageMaker

        # For this example, we'll simulate deployment by downloading the model
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Simulate preparing a serving environment
        serving_dir = "/tmp/model_serving"
        os.makedirs(serving_dir, exist_ok=True)

        # Download the model to the serving directory
        # In a real scenario, you might use MLflow's built-in serving capabilities
        loaded_model = mlflow.pyfunc.load_model(production_model_uri)

        # Create a simple prediction script
        prediction_script = f"""
import mlflow
import pandas as pd
import json
from flask import Flask, request, jsonify

# Load the model once at startup
model = mlflow.pyfunc.load_model("{production_model_uri}")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify({{"predictions": predictions.tolist()}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
"""

        with open(f"{serving_dir}/app.py", "w") as f:
            f.write(prediction_script)

        print(f"Model serving setup complete at {serving_dir}")

        # Push serving info to XCom
        kwargs["ti"].xcom_push(key="serving_dir", value=serving_dir)

        return {"serving_dir": serving_dir, "model_uri": production_model_uri}
    else:
        print("No model was promoted to production, skipping serving setup")
        return {"status": "skipped"}


def monitoring_setup(**kwargs):
    """Set up monitoring for the deployed model."""
    print("Setting up model monitoring...")

    # Pull data from XCom
    ti = kwargs["ti"]
    serving_result = ti.xcom_pull(task_ids="model_serving_setup")

    if serving_result and serving_result.get("serving_dir"):
        serving_dir = serving_result["serving_dir"]

        # In a real scenario, this would:
        # 1. Set up data quality monitoring
        # 2. Configure metrics collection for drift detection
        # 3. Set up alerts for performance degradation

        # For this example, we'll simulate by creating a monitoring script
        monitoring_script = """
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime
import mlflow

def collect_predictions(endpoint="http://localhost:8000/predict"):
    # Collect model predictions and actual outcomes
    return True

def calculate_metrics():
    # Calculate monitoring metrics
    return {
        "accuracy": 0.85,
        "data_drift_score": 0.05,
        "prediction_drift_score": 0.03
    }

def check_thresholds(metrics):
    # Check if metrics exceed alerting thresholds
    alerts = []
    if metrics["data_drift_score"] > 0.1:
        alerts.append("DATA_DRIFT_ALERT")
    if metrics["prediction_drift_score"] > 0.2:
        alerts.append("PREDICTION_DRIFT_ALERT")
    return alerts

def send_alert(alerts):
    # Send alerts via appropriate channels
    print(f"ALERTS: {alerts}")
    return True

def main():
    while True:
        collect_predictions()
        metrics = calculate_metrics()
        alerts = check_thresholds(metrics)
        
        if alerts:
            send_alert(alerts)
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_monitoring"):
            mlflow.log_metrics(metrics)
        
        time.sleep(3600)  # Run hourly

if __name__ == "__main__":
    main()
"""

        monitoring_path = f"{serving_dir}/monitoring.py"
        with open(monitoring_path, "w") as f:
            f.write(monitoring_script)

        print(f"Model monitoring setup complete at {monitoring_path}")

        # Push monitoring info to XCom
        kwargs["ti"].xcom_push(key="monitoring_path", value=monitoring_path)

        return {"monitoring_path": monitoring_path, "status": "complete"}
    else:
        print("No serving setup was completed, skipping monitoring setup")
        return {"status": "skipped"}


# Define the DAG tasks
t1_extract = PythonOperator(
    task_id="data_extraction",
    python_callable=data_extraction,
    provide_context=True,
    dag=dag,
)

t2_preprocess = PythonOperator(
    task_id="data_preprocessing",
    python_callable=data_preprocessing,
    provide_context=True,
    dag=dag,
)

t3_feature_eng = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
)

t4_split = PythonOperator(
    task_id="train_test_split_task",
    python_callable=train_test_split_task,
    provide_context=True,
    dag=dag,
)

t5_train = PythonOperator(
    task_id="model_training",
    python_callable=model_training,
    provide_context=True,
    dag=dag,
)

t6_evaluate = PythonOperator(
    task_id="model_evaluation",
    python_callable=model_evaluation,
    provide_context=True,
    dag=dag,
)

t7_promote = PythonOperator(
    task_id="model_promotion",
    python_callable=model_promotion,
    provide_context=True,
    dag=dag,
)

t8_serve = PythonOperator(
    task_id="model_serving_setup",
    python_callable=model_serving_setup,
    provide_context=True,
    dag=dag,
)

t9_monitor = PythonOperator(
    task_id="monitoring_setup",
    python_callable=monitoring_setup,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
(
    t1_extract
    >> t2_preprocess
    >> t3_feature_eng
    >> t4_split
    >> t5_train
    >> t6_evaluate
    >> t7_promote
    >> t8_serve
    >> t9_monitor
)
