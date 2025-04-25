# """
# Comprehensive Data Engineering Pipeline DAG
# This DAG demonstrates a complete data engineering workflow:
# - Data ingestion from a source
# - Data standardization
# - Data cleaning
# - Data quality validation using Great Expectations
# - Data transformation
# - Loading data to SQLite database
# """

# """
# This Airflow DAG implements a comprehensive data engineering pipeline with all the key components you requested. Here's a breakdown of what it includes:
# Key Features of the DAG:

# Complete Pipeline Stages:

# Ingestion: Reads from various file formats with a sample data generator
# Standardization: Normalizes column names and data types
# Cleaning: Handles duplicates, missing values, and outliers
# Quality Validation: Uses Great Expectations to verify data integrity
# Transformation: Creates new features and prepares data for loading
# Loading: Stores processed data in SQLite with configurable paths
# Analysis & Reporting: Provides summary statistics on processed data


# Great Expectations Integration:

# Validates data quality with multiple expectation checks
# Reports on validation results
# Flags records based on quality check results


# Configurable Parameters:

# Source file path
# SQLite database path
# Target table name
# Processing batch size
# Logging level


# Pandas Data Processing:

# Leverages pandas for all data manipulation tasks
# Handles data type conversion
# Performs feature engineering
# Manages data transformations


# Error Handling and Logging:

# Comprehensive logging throughout the pipeline
# Task-level exception handling
# Quality flags on processed data


# Metadata Enrichment:

# Adds processing timestamps
# Includes batch identifiers
# Maintains data quality flags


# To Use This DAG:

# Install required dependencies: airflow, pandas, great_expectations
# Place this DAG in your Airflow DAGs folder
# Configure the parameters when triggering the DAG:

# Set the source file path
# Specify the SQLite database location
# Adjust batch size and logging level as needed
# """

from datetime import datetime, timedelta
import os
import pandas as pd
import sqlite3
import logging
from io import StringIO

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Param
from airflow.utils.dates import days_ago

# Import Great Expectations libraries
import great_expectations as ge

# Default arguments for the DAG
default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email": ["data_engineer@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "comprehensive_data_pipeline",
    default_args=default_args,
    description="A comprehensive data engineering pipeline",
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=["data_pipeline", "data_engineering", "etl"],
    params={
        "source_file_path": Param("path/to/source/data.csv", type="string"),
        "sqlite_db_path": Param("/path/to/output/database.db", type="string"),
        "target_table": Param("processed_data", type="string"),
        "batch_size": Param(1000, type="integer"),
        "log_level": Param("INFO", type="string"),
    },
)


# Helper functions for logging and data management
def setup_logging(**kwargs):
    log_level = kwargs["params"]["log_level"]
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging setup complete")
    return "Logging setup complete"


def ingest_data(**kwargs):
    """
    Ingests data from a source file (CSV in this example)
    In a real scenario, this could connect to APIs, databases, etc.
    """
    source_path = kwargs["params"]["source_file_path"]
    logging.info(f"Ingesting data from {source_path}")

    try:
        # In a real scenario, you might use different methods based on the source
        if source_path.endswith(".csv"):
            df = pd.read_csv(source_path)
        elif source_path.endswith(".json"):
            df = pd.read_json(source_path)
        elif source_path.endswith(".xlsx") or source_path.endswith(".xls"):
            df = pd.read_excel(source_path)
        else:
            # For demonstration purposes, let's create sample data if file doesn't exist
            logging.warning(
                f"Source file not found or unsupported format. Creating sample data."
            )
            df = pd.DataFrame(
                {
                    "id": range(1, 101),
                    "name": [f"Product {i}" for i in range(1, 101)],
                    "category": ["A", "B", "C", "D"] * 25,
                    "price": [round(i * 1.5, 2) for i in range(1, 101)],
                    "date": pd.date_range(start="2023-01-01", periods=100),
                    "stock": [i * 10 for i in range(1, 101)],
                    "rating": [round(i % 5 + 0.5, 1) for i in range(1, 101)],
                }
            )

        logging.info(f"Ingested {len(df)} records")

        # Pass the dataframe to the next task
        kwargs["ti"].xcom_push(key="raw_data", value=df.to_json(orient="split"))
        return f"Data ingestion complete - {len(df)} records processed"

    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise


def standardize_data(**kwargs):
    """
    Standardizes the ingested data:
    - Convert column names to lowercase
    - Convert data types to appropriate formats
    - Handle date formats
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="ingest_data", key="raw_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Standardizing data")

    # Standardize column names
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

    # Convert data types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "stock" in df.columns:
        df["stock"] = df["stock"].astype(int, errors="ignore")

    # Add metadata column for tracking
    df["processed_timestamp"] = datetime.now()

    logging.info(f"Data standardization complete. Schema: {df.dtypes}")

    # Pass the standardized data to the next task
    kwargs["ti"].xcom_push(key="standardized_data", value=df.to_json(orient="split"))
    return "Data standardization complete"


def clean_data(**kwargs):
    """
    Cleans the standardized data:
    - Remove duplicates
    - Handle missing values
    - Remove outliers
    - Fix inconsistencies
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="standardize_data", key="standardized_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    initial_count = len(df)
    logging.info(f"Cleaning data. Initial record count: {initial_count}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_count - len(df)} duplicate records")

    # Handle missing values
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            logging.info(f"Found {missing_count} missing values in {column}")

            # Apply appropriate handling based on column type
            if pd.api.types.is_numeric_dtype(df[column]):
                # Fill numeric columns with median
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                logging.info(
                    f"Filled missing values in {column} with median: {median_value}"
                )
            elif pd.api.types.is_datetime64_dtype(df[column]):
                # Fill date columns with most recent date
                df[column].fillna(df[column].max(), inplace=True)
            else:
                # Fill other columns with mode or a placeholder
                if df[column].nunique() < 10:  # If low cardinality, use mode
                    mode_value = df[column].mode()[0]
                    df[column].fillna(mode_value, inplace=True)
                    logging.info(
                        f"Filled missing values in {column} with mode: {mode_value}"
                    )
                else:
                    # For high cardinality, use a placeholder
                    df[column].fillna("UNKNOWN", inplace=True)
                    logging.info(f"Filled missing values in {column} with 'UNKNOWN'")

    # Handle outliers in numeric columns
    for column in df.select_dtypes(include=["number"]).columns:
        if column in ["id", "stock", "processed_timestamp"]:  # Skip certain columns
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        if outliers > 0:
            logging.info(f"Found {outliers} outliers in {column}")
            # Instead of removing, cap the outliers
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            logging.info(
                f"Capped outliers in {column} to range: [{lower_bound}, {upper_bound}]"
            )

    # Add data quality flag
    df["data_quality_flag"] = "CLEAN"

    logging.info(f"Data cleaning complete. Final record count: {len(df)}")

    # Pass the cleaned data to the next task
    kwargs["ti"].xcom_push(key="cleaned_data", value=df.to_json(orient="split"))
    return f"Data cleaning complete - {len(df)} clean records"


def validate_data_quality(**kwargs):
    """
    Validates data quality using Great Expectations
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="clean_data", key="cleaned_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Validating data quality with Great Expectations")

    # Create a GE DataFrame
    ge_df = ge.from_pandas(df)

    # Define and run expectations
    expectations_results = []

    # Expectation 1: Check for missing values
    result = ge_df.expect_column_values_to_not_be_null("id")
    expectations_results.append(("No nulls in ID", result.success))

    # Expectation 2: Check value distributions if 'category' exists
    if "category" in df.columns:
        result = ge_df.expect_column_values_to_be_in_set(
            "category", ["A", "B", "C", "D"]
        )
        expectations_results.append(("Categories are valid", result.success))

    # Expectation 3: Check numeric ranges if 'price' exists
    if "price" in df.columns:
        result = ge_df.expect_column_values_to_be_between("price", 0, 1000)
        expectations_results.append(("Price is in valid range", result.success))

    # Expectation 4: Check data types
    for col in df.select_dtypes(include=["number"]).columns:
        result = ge_df.expect_column_values_to_be_of_type(col, "float64", "numpy")
        expectations_results.append((f"{col} is numeric", result.success))

    # Expectation 5: Check for uniqueness
    result = ge_df.expect_column_values_to_be_unique("id")
    expectations_results.append(("IDs are unique", result.success))

    # Log results and add quality flags
    all_passed = all(result[1] for result in expectations_results)
    for name, success in expectations_results:
        status = "PASSED" if success else "FAILED"
        logging.info(f"Data quality check '{name}': {status}")

    # Update quality flags based on validation results
    if not all_passed:
        df.loc[~df["id"].duplicated(), "data_quality_flag"] = "WARNING"
        logging.warning(
            "Some data quality checks failed. Records flagged with WARNING."
        )

    # Pass the validated data to the next task
    kwargs["ti"].xcom_push(key="validated_data", value=df.to_json(orient="split"))

    return f"Data quality validation complete - {'All checks passed' if all_passed else 'Some checks failed'}"


def transform_data(**kwargs):
    """
    Transforms the data to derive new features and prepare for loading
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="validate_data_quality", key="validated_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Transforming data")

    # Feature engineering
    if all(col in df.columns for col in ["price", "stock"]):
        # Calculate inventory value
        df["inventory_value"] = df["price"] * df["stock"]
        logging.info("Added 'inventory_value' feature")

    if "date" in df.columns:
        # Extract date components
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        logging.info("Added date-based features")

    if "category" in df.columns:
        # One-hot encode categorical data
        category_dummies = pd.get_dummies(df["category"], prefix="category")
        df = pd.concat([df, category_dummies], axis=1)
        logging.info(
            f"One-hot encoded 'category' into {len(category_dummies.columns)} features"
        )

    # Add process metadata
    df["etl_timestamp"] = datetime.now()
    df["etl_batch_id"] = datetime.now().strftime("%Y%m%d%H%M%S")

    logging.info(f"Data transformation complete. Final schema: {list(df.columns)}")

    # Pass the transformed data to the next task
    kwargs["ti"].xcom_push(key="transformed_data", value=df.to_json(orient="split"))
    return "Data transformation complete"


def load_data(**kwargs):
    """
    Loads transformed data into SQLite database
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="transform_data", key="transformed_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    sqlite_db_path = kwargs["params"]["sqlite_db_path"]
    target_table = kwargs["params"]["target_table"]
    batch_size = kwargs["params"]["batch_size"]

    logging.info(
        f"Loading data to SQLite database: {sqlite_db_path}, table: {target_table}"
    )

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)

        # Create connection to SQLite database
        conn = sqlite3.connect(sqlite_db_path)

        # Load data in batches to avoid memory issues
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i : min(i + batch_size, total_rows)]
            batch_df.to_sql(
                target_table,
                conn,
                if_exists="append" if i > 0 else "replace",
                index=False,
            )
            logging.info(
                f"Loaded batch {i // batch_size + 1}/{(total_rows - 1) // batch_size + 1} ({len(batch_df)} records)"
            )

        # Verify the data was loaded correctly
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {target_table}")
        db_count = cursor.fetchone()[0]

        if db_count == total_rows:
            logging.info(f"Successfully loaded {db_count} records to database")
        else:
            logging.warning(
                f"Data count mismatch! Expected {total_rows}, found {db_count} in database"
            )

        # Close connection
        conn.close()

        return f"Data loading complete - {db_count} records loaded to {target_table}"

    except Exception as e:
        logging.error(f"Error during data loading: {str(e)}")
        raise


def analyze_and_report(**kwargs):
    """
    Performs a simple analysis on the data and generates a report
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="transform_data", key="transformed_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Generating analysis report")

    # Generate summary statistics
    summary = {}
    for col in df.select_dtypes(include=["number"]).columns:
        if col not in ["etl_timestamp", "id", "year", "month", "day_of_week"]:
            summary[col] = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median(),
            }

    # Count by category if it exists
    if "category" in df.columns:
        category_counts = df["category"].value_counts().to_dict()
        summary["category_distribution"] = category_counts

    # Summary report
    report = "Data Pipeline Analysis Report\n"
    report += "=" * 30 + "\n"
    report += f"Total records processed: {len(df)}\n"
    report += f"Processing timestamp: {datetime.now()}\n"
    report += "=" * 30 + "\n"

    report += "\nNumeric Feature Statistics:\n"
    for col, stats in summary.items():
        if col != "category_distribution":
            report += f"\n{col}:\n"
            for stat_name, value in stats.items():
                report += f"  - {stat_name}: {value}\n"

    if "category_distribution" in summary:
        report += "\nCategory Distribution:\n"
        for category, count in summary["category_distribution"].items():
            report += f"  - {category}: {count} records\n"

    logging.info("Analysis report generated")
    logging.info(report)

    # In a real scenario, you might save this report to a file, send an email, etc.
    return "Analysis and reporting complete"


# Define task dependencies
task_setup_logging = PythonOperator(
    task_id="setup_logging",
    python_callable=setup_logging,
    provide_context=True,
    dag=dag,
)

task_ingest_data = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_data,
    provide_context=True,
    dag=dag,
)

task_standardize_data = PythonOperator(
    task_id="standardize_data",
    python_callable=standardize_data,
    provide_context=True,
    dag=dag,
)

task_clean_data = PythonOperator(
    task_id="clean_data",
    python_callable=clean_data,
    provide_context=True,
    dag=dag,
)

task_validate_data_quality = PythonOperator(
    task_id="validate_data_quality",
    python_callable=validate_data_quality,
    provide_context=True,
    dag=dag,
)

task_transform_data = PythonOperator(
    task_id="transform_data",
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

task_load_data = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

task_analyze_and_report = PythonOperator(
    task_id="analyze_and_report",
    python_callable=analyze_and_report,
    provide_context=True,
    dag=dag,
)

# Set up the task dependencies
(
    task_setup_logging
    >> task_ingest_data
    >> task_standardize_data
    >> task_clean_data
    >> task_validate_data_quality
    >> task_transform_data
    >> task_load_data
    >> task_analyze_and_report
)
