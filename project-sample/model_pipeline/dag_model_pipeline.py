# model engineering pipeline DAGs

from model_pipeline.model_engineering import train_model, evaluate_model
from model_pipeline.model_deployment import register_model
from model_pipeline.model_serving import serve_model, load_model
from airflow import DAG
from airflow.decorators import task
from datetime import timedelta
from datetime import datetime


params = {
    "model_name": "my_model",
    "model_uri": "models:/my_model/1",
    "input_data": {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    },
    "output_data": {
        "prediction": [0.5, 0.6, 0.7],
    },
}

with DAG(
    "model_engineering_pipeline",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    params=params,
    tags=["model_pipeline"],
    description="Model engineering pipeline DAG",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 4, 1),
    catchup=False,
) as dag:
    # Define the tasks
    @task
    def train_model(**context):
        """
        Train a machine learning model.
        Args:
            context (dict): Airflow context containing parameters.
        """
        # Get parameters from the context
        model_name = context["dag_run"].conf.get("model_name")
        input_data = context["dag_run"].conf.get("input_data")
        # Train the model
        train_model(model_name, input_data)

    @task
    def evaluate_model(**context):
        """
        Evaluate the trained model.
        Args:
            context (dict): Airflow context containing parameters.
        """
        # Get parameters from the context
        model_name = context["dag_run"].conf.get("model_name")
        input_data = context["dag_run"].conf.get("input_data")
        output_data = context["dag_run"].conf.get("output_data")
        # Evaluate the model
        evaluate_model(model_name, input_data, output_data)

    @task
    def register_model(**context):
        """
        Register the trained model in MLflow.
        Args:
            context (dict): Airflow context containing parameters.
        """
        # Get parameters from the context
        model_name = context["dag_run"].conf.get("model_name")
        model_uri = context["dag_run"].conf.get("model_uri")
        # Register the model
        register_model(model_name, model_uri)

    @task
    def load_model(**context):
        """
        Load a model from MLflow.
        Args:
            context (dict): Airflow context containing parameters.
        """
        # Get parameters from the context
        model_uri = context["dag_run"].conf.get("model_uri")
        # Load the model
        load_model(model_uri)

    @task
    def serve_model(**context):
        """
        Serve a model using MLflow.
        Args:
            context (dict): Airflow context containing parameters.
        """
        # Get parameters from the context
        model_uri = context["dag_run"].conf.get("model_uri")
        input_data = context["dag_run"].conf.get("input_data")
        output_data = context["dag_run"].conf.get("output_data")
        # Serve the model
        serve_model(model_uri, input_data, output_data)

    # Define the task dependencies
    train_model_task = train_model()
    evaluate_model_task = evaluate_model()
    register_model_task = register_model()
    load_model_task = load_model()
    serve_model_task = serve_model()
    (
        train_model_task
        >> evaluate_model_task
        >> register_model_task
        >> load_model_task
        >> serve_model_task
    )
