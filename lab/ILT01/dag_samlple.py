from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator

# Import necessary libraries
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define functions for each task

def read_data(**kwargs):
    # Read data from online NYC dataset
    url = "https://example.com/nyc_dataset.csv"  # Replace with actual URL
    data = pd.read_csv(url)
    data.to_csv('nyc_data.csv', index=False)

def check_data_quality(**kwargs):
    # Use Great Expectations to check data quality
    df = pd.read_csv('nyc_data.csv')
    ge_df = ge.from_pandas(df)
    expectation_suite = ge ExpectationSuite(expectation_suite_name="data_quality")
    expectation_suite.add_expectation(
        ge.expect_column_values_to_not_be_null(column="column_name")  # Replace with actual column name
    )
    results = ge_df.validate(expectation_suite)
    if results["success"]:
        return "quality_passed"
    else:
        return "quality_failed"
    
def decide_next_task(**kwargs):
    quality_result = kwargs['ti'].xcom_pull(task_ids='check_data_quality')
    if quality_result == "quality_passed":
        return "train_model"
    else:
        return "clean_data"


def clean_data(**kwargs):
    # Clean data using Pandas
    df = pd.read_csv('nyc_data.csv')
    # Perform cleaning operations here
    df.to_csv('cleaned_nyc_data.csv', index=False)

def train_model(**kwargs):
    # Train a simple ML model
    df = pd.read_csv('nyc_data.csv')
    # Prepare data for training
    X = df.drop('target_column', axis=1)  # Replace with actual target column
    y = df['target_column']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save the model
    import pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

def deploy_model(**kwargs):
    # Deploy the trained model
    # Implement model deployment logic here
    pass

def monitor_model(**kwargs):
    # Monitor the deployed model's performance
    # Implement model monitoring logic here
    pass

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'nyc_data_pipeline',
    default_args=default_args,
    description='A data pipeline for NYC dataset',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
) as dag:
    # Define tasks
    read_data_task = PythonOperator(
        task_id='read_data',
        python_callable=read_data
    )

    check_quality_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        trigger_rule='all_failed'
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        trigger_rule='all_done'
    )

    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model
    )

    monitor_model_task = PythonOperator(
        task_id='monitor_model',
        python_callable=monitor_model
    )
    
	# Define the branch task
    branch_task = BranchPythonOperator(
        task_id='decide_next_task',
        python_callable=decide_next_task
    )


    # Define dependencies
    read_data_task >> check_quality_task
    check_quality_task >> branch_task
    branch_task >> [clean_data_task, train_model_task]
    clean_data_task >> train_model_task
    train_model_task >> deploy_model_task
    deploy_model_task >> monitor_model_task

