# model engineering pipeline DAGs

from model_pipeline.model_engineering import train_model, evaluate_model
from model_pipeline.model_deployment import register_model
from model_pipeline.model_serving import serve_model, load_model
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from datetime import datetime

# from airflow.decorators import task

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

# import os
# import sys
# Set up the virtual environment for the DAG
# Activate the virtual environment
# venv_path = "/path/to/your/virtualenv"
# activate_this = os.path.join(venv_path, "bin", "activate_this.py")
# with open(activate_this) as f:
#     exec(f.read(), {"__file__": activate_this})

# # Add the virtual environment's site-packages to sys.path
# venv_site_packages = os.path.join(venv_path, "lib", "python3.x", "site-packages")
# sys.path.insert(0, venv_site_packages)

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
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "model_name": "{{ params.model_name }}",
            "input_data": "{{ params.input_data }}",
        },
        requirements=["pandas", "scikit-learn", "mlflow"],
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "model_name": "{{ params.model_name }}",
            "input_data": "{{ params.input_data }}",
            "output_data": "{{ params.output_data }}",
        },
        requirements=["pandas", "scikit-learn", "mlflow"],
    )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        op_kwargs={
            "model_name": "{{ params.model_name }}",
            "model_uri": "{{ params.model_uri }}",
        },
        requirements=["pandas", "scikit-learn", "mlflow"],
    )

    load_model_task = PythonOperator(
        task_id="load_model",
        python_callable=load_model,
        op_kwargs={
            "model_uri": "{{ params.model_uri }}",
        },
        requirements=["pandas", "scikit-learn", "mlflow"],
    )

    serve_model_task = PythonOperator(
        task_id="serve_model",
        python_callable=serve_model,
        op_kwargs={
            "model_uri": "{{ params.model_uri }}",
            "input_data": "{{ params.input_data }}",
            "output_data": "{{ params.output_data }}",
        },
        requirements=["pandas", "scikit-learn", "mlflow"],
    )

    # Define the task dependencies
    (
        train_model_task
        >> evaluate_model_task
        >> register_model_task
        >> load_model_task
        >> serve_model_task
    )
