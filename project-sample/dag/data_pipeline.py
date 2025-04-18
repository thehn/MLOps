# data_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task

from data_pipeline.ingestion import ingest_data
from data_pipeline.processing import validate_table
from data_pipeline.feature_engineering import engineering_features


# Define the parameters for the DAG
params = {
    "raw_csv_file": "/path/to/raw_data.csv",
    "golden_database": "/path/to/golden_data.db",
    "feature_store_database": "/path/to/feature_store.db",
    "table_name": "data_table",
    "golden_table_name": "data_table",
    "feature_store_table_name": "features_table",
}

# Set the default parameters for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "data_engineering_pipeline",
    default_args=default_args,
    params=params,
    tags=["data_pipeline"],
    description="Data engineering pipeline DAG",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 4, 1),
    catchup=False,
) as dag:

    @task
    def task_ingest_data(**context):
        raw_csv_file = context["dag_run"].conf.get("raw_csv_file")
        golden_database = context["dag_run"].conf.get("golden_database")
        table_name = context["dag_run"].conf.get("table_name", "data_table")

        ingest_data(raw_csv_file, golden_database, table_name)

    @task
    def task_validate_table(**context):
        golden_database = context["dag_run"].conf.get("golden_database")
        table_name = context["dag_run"].conf.get("table_name", "data_table")

        validate_table(golden_database, table_name)

    @task
    def task_engineer_features(**context):
        golden_database = context["dag_run"].conf.get("golden_database")
        feature_store_database = context["dag_run"].conf.get("feature_store_database")
        golden_table_name = context["dag_run"].conf.get(
            "golden_table_name", "data_table"
        )
        feature_store_table_name = context["dag_run"].conf.get(
            "feature_store_table_name", "features_table"
        )

        engineering_features(
            golden_database,
            golden_table_name,
            feature_store_database,
            feature_store_table_name,
        )

    # Define the task dependencies
    ingest_data_task = task_ingest_data()
    validate_table_task = task_validate_table()
    engineer_features_task = task_engineer_features()
    ingest_data_task >> validate_table_task >> engineer_features_task
