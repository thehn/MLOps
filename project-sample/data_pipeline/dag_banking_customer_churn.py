"""
Banking Customer Data Engineering Pipeline DAG
This DAG processes banking customer data with the following workflow:
- Data ingestion from CSV
- Data standardization and type conversion
- Data cleaning and handling missing values
- Data quality validation using Great Expectations
- Feature engineering and transformation
- Loading data to SQLite database
- Analysis and reporting
"""

from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
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
    "banking_customer_data_pipeline",
    default_args=default_args,
    description="Banking customer data engineering pipeline",
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=["data_engineering", "etl", "banking"],
    params={
        "source_file_path": Param("path/to/banking_data.csv", type="string"),
        "sqlite_db_path": Param("/path/to/output/banking_customers.db", type="string"),
        "target_table": Param("customer_analytics", type="string"),
        "batch_size": Param(1000, type="integer"),
        "log_level": Param("INFO", type="string"),
    },
)

# Define column data types based on sample data
COLUMN_DTYPES = {
    "RowNumber": "int64",
    "CustomerId": "int64",
    "Surname": "string",
    "CreditScore": "int64",
    "Geography": "category",
    "Gender": "category",
    "Age": "int64",
    "Tenure": "int64",
    "Balance": "float64",
    "NumOfProducts": "int64",
    "HasCrCard": "int64",
    "IsActiveMember": "int64",
    "EstimatedSalary": "float64",
    "Exited": "int64",
}

# Define categorical variables
CATEGORICAL_VARS = ["Geography", "Gender"]

# Define binary variables
BINARY_VARS = ["HasCrCard", "IsActiveMember", "Exited"]


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
    Ingests banking customer data from the CSV file
    """
    source_path = kwargs["params"]["source_file_path"]
    logging.info(f"Ingesting banking customer data from {source_path}")

    try:
        # Check if file exists
        if os.path.exists(source_path):
            # Read CSV with appropriate data types
            df = pd.read_csv(
                source_path,
                dtype={
                    "CustomerId": str,  # Store ID as string to preserve leading zeros
                    "Geography": "category",
                    "Gender": "category",
                },
            )
            logging.info(f"Successfully loaded data from {source_path}")
        else:
            # For testing: Create sample data similar to the provided schema if file doesn't exist
            logging.warning(
                f"Source file not found: {source_path}. Creating sample data for testing."
            )

            # Create sample data based on the provided sample records
            df = pd.DataFrame(
                {
                    "RowNumber": range(1, 101),
                    "CustomerId": [f"{15600000 + i}" for i in range(1, 101)],
                    "Surname": ["Smith", "Johnson", "Brown", "Lee", "Garcia"] * 20,
                    "CreditScore": np.random.randint(300, 900, 100),
                    "Geography": np.random.choice(["France", "Spain", "Germany"], 100),
                    "Gender": np.random.choice(["Male", "Female"], 100),
                    "Age": np.random.randint(18, 95, 100),
                    "Tenure": np.random.randint(0, 11, 100),
                    "Balance": np.random.uniform(0, 250000, 100),
                    "NumOfProducts": np.random.randint(1, 5, 100),
                    "HasCrCard": np.random.randint(0, 2, 100),
                    "IsActiveMember": np.random.randint(0, 2, 100),
                    "EstimatedSalary": np.random.uniform(10000, 200000, 100),
                    "Exited": np.random.randint(0, 2, 100),
                }
            )

            # Set appropriate data types
            df["Geography"] = df["Geography"].astype("category")
            df["Gender"] = df["Gender"].astype("category")

        logging.info(f"Ingested {len(df)} customer records")

        # Display initial data profile
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Column data types: {df.dtypes}")

        # Pass the dataframe to the next task
        kwargs["ti"].xcom_push(key="raw_data", value=df.to_json(orient="split"))
        return f"Data ingestion complete - {len(df)} customer records processed"

    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")
        raise


def standardize_data(**kwargs):
    """
    Standardizes the banking data:
    - Ensures consistent data types
    - Converts categorical data
    - Standardizes column names
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="ingest_data", key="raw_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Standardizing banking customer data")

    # 1. Standardize column names (already properly named in this case)
    # Keep original column names for this dataset as they are already in good format

    # 2. Convert data types according to our schema definition
    # Handle numeric columns
    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            if dtype in ["int64", "float64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "category" and col in CATEGORICAL_VARS:
                df[col] = df[col].astype("category")

    # 3. Ensure binary variables are 0 or 1
    for col in BINARY_VARS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 4. Add metadata column for tracking
    df["processed_timestamp"] = datetime.now()

    # Log standardization results
    logging.info(f"Data standardization complete. Schema: {df.dtypes}")

    # Pass the standardized data to the next task
    kwargs["ti"].xcom_push(key="standardized_data", value=df.to_json(orient="split"))
    return "Banking data standardization complete"


def clean_data(**kwargs):
    """
    Cleans the banking customer data:
    - Handle missing values
    - Remove outliers
    - Handle inconsistencies
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="standardize_data", key="standardized_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    initial_count = len(df)
    logging.info(f"Cleaning banking data. Initial record count: {initial_count}")

    # 1. Remove duplicates based on CustomerId (should be unique)
    df_unique = df.drop_duplicates(subset=["CustomerId"])
    if len(df_unique) < initial_count:
        logging.warning(
            f"Removed {initial_count - len(df_unique)} duplicate customer records"
        )
        df = df_unique

    # 2. Handle missing values
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

    for column in columns_with_missing:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            logging.info(f"Found {missing_count} missing values in {column}")

            # Apply appropriate handling based on column type
            if column in ["CreditScore", "Age", "Tenure", "NumOfProducts"]:
                # For important customer attributes, use median
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                logging.info(
                    f"Filled missing values in {column} with median: {median_value}"
                )

            elif column in ["Balance", "EstimatedSalary"]:
                # For financial values, use median
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                logging.info(
                    f"Filled missing values in {column} with median: {median_value}"
                )

            elif column in CATEGORICAL_VARS:
                # For categorical, use mode
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
                logging.info(
                    f"Filled missing values in {column} with mode: {mode_value}"
                )

            elif column in BINARY_VARS:
                # For binary columns, use 0 (most conservative approach)
                df[column].fillna(0, inplace=True)
                logging.info(f"Filled missing values in {column} with 0")

            else:
                # For other columns, use appropriate strategy
                if df[column].dtype == "object":
                    df[column].fillna("UNKNOWN", inplace=True)
                else:
                    df[column].fillna(df[column].median(), inplace=True)

    # 3. Handle outliers in relevant numeric columns
    outlier_columns = ["Age", "CreditScore", "Balance", "EstimatedSalary"]

    for column in outlier_columns:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers - don't drop but cap them to preserve data
            outliers_count = (
                (df[column] < lower_bound) | (df[column] > upper_bound)
            ).sum()
            if outliers_count > 0:
                logging.info(f"Found {outliers_count} outliers in {column}")
                # Cap the outliers instead of removing
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                logging.info(
                    f"Capped outliers in {column} to range: [{lower_bound}, {upper_bound}]"
                )

    # 4. Ensure data integrity constraints
    # Age should be reasonable for banking customers
    if "Age" in df.columns:
        invalid_age = (df["Age"] < 18).sum()
        if invalid_age > 0:
            logging.warning(f"Found {invalid_age} records with age < 18, setting to 18")
            df.loc[df["Age"] < 18, "Age"] = 18

    # 5. Add data quality flag
    df["data_quality_flag"] = "CLEAN"

    logging.info(f"Data cleaning complete. Final record count: {len(df)}")

    # Pass the cleaned data to the next task
    kwargs["ti"].xcom_push(key="cleaned_data", value=df.to_json(orient="split"))
    return f"Banking data cleaning complete - {len(df)} clean records"


def validate_data_quality(**kwargs):
    """
    Validates banking data quality using Great Expectations
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="clean_data", key="cleaned_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Validating banking data quality with Great Expectations")

    # Create a GE DataFrame
    ge_df = ge.from_pandas(df)

    # Define and run expectations
    expectations_results = []

    # Expectation 1: Check for missing values in key columns
    key_columns = ["CustomerId", "CreditScore", "Age", "Balance"]
    for col in key_columns:
        if col in df.columns:
            result = ge_df.expect_column_values_to_not_be_null(col)
            expectations_results.append((f"No nulls in {col}", result.success))

    # Expectation 2: Check for valid Geography values
    if "Geography" in df.columns:
        valid_countries = ["France", "Spain", "Germany"]
        result = ge_df.expect_column_values_to_be_in_set("Geography", valid_countries)
        expectations_results.append(("Geography values are valid", result.success))

    # Expectation 3: Check for valid Gender values
    if "Gender" in df.columns:
        result = ge_df.expect_column_values_to_be_in_set("Gender", ["Male", "Female"])
        expectations_results.append(("Gender values are valid", result.success))

    # Expectation 4: Check range for Age
    if "Age" in df.columns:
        result = ge_df.expect_column_values_to_be_between("Age", 18, 100)
        expectations_results.append(("Age is in valid range", result.success))

    # Expectation 5: Check range for CreditScore
    if "CreditScore" in df.columns:
        result = ge_df.expect_column_values_to_be_between("CreditScore", 300, 900)
        expectations_results.append(("CreditScore is in valid range", result.success))

    # Expectation 6: Check binary columns contain only 0 or 1
    for col in BINARY_VARS:
        if col in df.columns:
            result = ge_df.expect_column_values_to_be_in_set(col, [0, 1])
            expectations_results.append((f"{col} contains only 0 or 1", result.success))

    # Expectation 7: Check for uniqueness in CustomerId
    if "CustomerId" in df.columns:
        result = ge_df.expect_column_values_to_be_unique("CustomerId")
        expectations_results.append(("CustomerIds are unique", result.success))

    # Log results and add quality flags
    all_passed = all(result[1] for result in expectations_results)
    for name, success in expectations_results:
        status = "PASSED" if success else "FAILED"
        logging.info(f"Data quality check '{name}': {status}")

    # Update quality flags based on validation results
    if not all_passed:
        failing_expectations = [
            result[0] for result in expectations_results if not result[1]
        ]
        df["data_quality_flag"] = "WARNING"
        logging.warning(f"Failed quality checks: {', '.join(failing_expectations)}")

    # Pass the validated data to the next task
    kwargs["ti"].xcom_push(key="validated_data", value=df.to_json(orient="split"))

    return f"Banking data quality validation complete - {'All checks passed' if all_passed else 'Some checks failed'}"


def transform_data(**kwargs):
    """
    Transforms the banking data to derive new features for analysis
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="validate_data_quality", key="validated_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Transforming banking customer data")

    # 1. Feature Engineering

    # Age groups / segments
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 30, 40, 50, 60, 100],
            labels=["Under 30", "30-40", "40-50", "50-60", "Over 60"],
        )
        logging.info("Added AgeGroup feature")

    # Credit Score segments
    if "CreditScore" in df.columns:
        df["CreditScoreGroup"] = pd.cut(
            df["CreditScore"],
            bins=[0, 580, 670, 740, 800, 900],
            labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        )
        logging.info("Added CreditScoreGroup feature")

    # Calculate Balance-to-Salary ratio (financial health indicator)
    if all(col in df.columns for col in ["Balance", "EstimatedSalary"]):
        # Avoid division by zero
        df["BalanceToSalary"] = df.apply(
            lambda row: row["Balance"] / row["EstimatedSalary"]
            if row["EstimatedSalary"] > 0
            else 0,
            axis=1,
        )
        logging.info("Added BalanceToSalary feature")

    # Customer activity score (composite metric)
    if all(
        col in df.columns for col in ["IsActiveMember", "NumOfProducts", "HasCrCard"]
    ):
        df["ActivityScore"] = (
            df["IsActiveMember"] * 5 + df["NumOfProducts"] + df["HasCrCard"]
        )
        logging.info("Added ActivityScore feature")

    # One-hot encode categorical variables
    categorical_columns = [col for col in CATEGORICAL_VARS if col in df.columns]
    if categorical_columns:
        df_encoded = pd.get_dummies(
            df, columns=categorical_columns, prefix=categorical_columns
        )
        df = df_encoded
        logging.info(
            f"One-hot encoded {len(categorical_columns)} categorical variables"
        )

    # 2. Add process metadata
    df["etl_timestamp"] = datetime.now()
    df["etl_batch_id"] = datetime.now().strftime("%Y%m%d%H%M%S")

    logging.info(
        f"Banking data transformation complete. Final schema: {list(df.columns)}"
    )

    # Pass the transformed data to the next task
    kwargs["ti"].xcom_push(key="transformed_data", value=df.to_json(orient="split"))
    return "Banking data transformation complete"


def load_data(**kwargs):
    """
    Loads transformed banking data into SQLite database
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="transform_data", key="transformed_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    sqlite_db_path = kwargs["params"]["sqlite_db_path"]
    target_table = kwargs["params"]["target_table"]
    batch_size = kwargs["params"]["batch_size"]

    logging.info(
        f"Loading banking data to SQLite database: {sqlite_db_path}, table: {target_table}"
    )

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)

        # Create connection to SQLite database
        conn = sqlite3.connect(sqlite_db_path)

        # Convert categorical columns to string before saving to SQLite
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype(str)

        # Load data in batches to avoid memory issues
        total_rows = len(df)
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i : min(i + batch_size, total_rows)]

            # For the first batch, replace the table, then append
            if_exists_param = "replace" if i == 0 else "append"

            batch_df.to_sql(target_table, conn, if_exists=if_exists_param, index=False)
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

        return f"Banking data loading complete - {db_count} records loaded to {target_table}"

    except Exception as e:
        logging.error(f"Error during data loading: {str(e)}")
        raise


def analyze_and_report(**kwargs):
    """
    Performs analysis on banking customer data and generates a report
    """
    ti = kwargs["ti"]
    data_json = ti.xcom_pull(task_ids="transform_data", key="transformed_data")
    df = pd.read_json(StringIO(data_json), orient="split")

    logging.info("Generating banking customer analysis report")

    # 1. Customer Demographics Analysis
    demographics = {}

    # Geography distribution
    if "Geography" in df.columns:
        geography_counts = df["Geography"].value_counts().to_dict()
        demographics["geography_distribution"] = geography_counts

    # Gender distribution
    if "Gender" in df.columns:
        gender_counts = df["Gender"].value_counts().to_dict()
        demographics["gender_distribution"] = gender_counts

    # Age group distribution
    if "AgeGroup" in df.columns:
        age_counts = df["AgeGroup"].value_counts().to_dict()
        demographics["age_distribution"] = age_counts

    # 2. Financial Analysis
    financial = {}

    # Credit score statistics
    if "CreditScore" in df.columns:
        financial["credit_score"] = {
            "min": df["CreditScore"].min(),
            "max": df["CreditScore"].max(),
            "mean": df["CreditScore"].mean(),
            "median": df["CreditScore"].median(),
        }

    # Balance statistics
    if "Balance" in df.columns:
        financial["balance"] = {
            "min": df["Balance"].min(),
            "max": df["Balance"].max(),
            "mean": df["Balance"].mean(),
            "median": df["Balance"].median(),
            "zero_balance_count": (df["Balance"] == 0).sum(),
        }

    # 3. Customer Behavior Analysis
    behavior = {}

    # Active members vs inactive
    if "IsActiveMember" in df.columns:
        active_count = df["IsActiveMember"].sum()
        behavior["active_members"] = active_count
        behavior["inactive_members"] = len(df) - active_count
        behavior["active_percentage"] = round((active_count / len(df)) * 100, 2)

    # Product distribution
    if "NumOfProducts" in df.columns:
        product_counts = df["NumOfProducts"].value_counts().to_dict()
        behavior["product_distribution"] = product_counts

    # Credit card ownership
    if "HasCrCard" in df.columns:
        has_card_count = df["HasCrCard"].sum()
        behavior["has_credit_card"] = has_card_count
        behavior["credit_card_percentage"] = round((has_card_count / len(df)) * 100, 2)

    # 4. Churn Analysis
    churn = {}

    if "Exited" in df.columns:
        churn_count = df["Exited"].sum()
        churn["churned_customers"] = churn_count
        churn["retained_customers"] = len(df) - churn_count
        churn["churn_rate"] = round((churn_count / len(df)) * 100, 2)

        # Churn by geography
        if "Geography" in df.columns:
            churn_by_geo = df.groupby("Geography")["Exited"].mean().round(4) * 100
            churn["churn_by_geography"] = churn_by_geo.to_dict()

        # Churn by age group
        if "AgeGroup" in df.columns:
            churn_by_age = df.groupby("AgeGroup")["Exited"].mean().round(4) * 100
            churn["churn_by_age"] = churn_by_age.to_dict()

    # 5. Format the report
    report = "Banking Customer Data Analysis Report\n"
    report += "=" * 40 + "\n"
    report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"Total Customers Analyzed: {len(df)}\n"
    report += "=" * 40 + "\n\n"

    # Demographics section
    report += "1. CUSTOMER DEMOGRAPHICS\n"
    report += "-" * 30 + "\n"

    if "geography_distribution" in demographics:
        report += "\nGeography Distribution:\n"
        for country, count in demographics["geography_distribution"].items():
            report += f"  - {country}: {count} customers ({round(count / len(df) * 100, 1)}%)\n"

    if "gender_distribution" in demographics:
        report += "\nGender Distribution:\n"
        for gender, count in demographics["gender_distribution"].items():
            report += f"  - {gender}: {count} customers ({round(count / len(df) * 100, 1)}%)\n"

    if "age_distribution" in demographics:
        report += "\nAge Distribution:\n"
        for age_group, count in demographics["age_distribution"].items():
            report += f"  - {age_group}: {count} customers ({round(count / len(df) * 100, 1)}%)\n"

    # Financial section
    report += "\n\n2. FINANCIAL ANALYSIS\n"
    report += "-" * 30 + "\n"

    if "credit_score" in financial:
        report += "\nCredit Score Statistics:\n"
        for metric, value in financial["credit_score"].items():
            report += f"  - {metric.capitalize()}: {round(value, 2)}\n"

    if "balance" in financial:
        report += "\nBalance Statistics:\n"
        for metric, value in financial["balance"].items():
            if metric != "zero_balance_count":
                report += f"  - {metric.capitalize()}: {round(value, 2)}\n"
        report += f"  - Customers with zero balance: {financial['balance']['zero_balance_count']} "
        report += f"({round(financial['balance']['zero_balance_count'] / len(df) * 100, 1)}%)\n"

    # Behavior section
    report += "\n\n3. CUSTOMER BEHAVIOR\n"
    report += "-" * 30 + "\n"

    if "active_members" in behavior:
        report += f"\nActive Members: {behavior['active_members']} ({behavior['active_percentage']}%)\n"
        report += f"Inactive Members: {behavior['inactive_members']} ({100 - behavior['active_percentage']}%)\n"

    if "product_distribution" in behavior:
        report += "\nProduct Distribution:\n"
        for num_products, count in behavior["product_distribution"].items():
            report += f"  - {num_products} product(s): {count} customers ({round(count / len(df) * 100, 1)}%)\n"

    if "has_credit_card" in behavior:
        report += f"\nCredit Card Ownership: {behavior['has_credit_card']} customers ({behavior['credit_card_percentage']}%)\n"

    # Churn section
    report += "\n\n4. CUSTOMER CHURN ANALYSIS\n"
    report += "-" * 30 + "\n"

    if "churned_customers" in churn:
        report += f"\nOverall Churn Rate: {churn['churn_rate']}%\n"
        report += f"Churned Customers: {churn['churned_customers']}\n"
        report += f"Retained Customers: {churn['retained_customers']}\n"

        if "churn_by_geography" in churn:
            report += "\nChurn Rate by Geography:\n"
            for geo, rate in churn["churn_by_geography"].items():
                report += f"  - {geo}: {round(rate, 2)}%\n"

        if "churn_by_age" in churn:
            report += "\nChurn Rate by Age Group:\n"
            for age, rate in churn["churn_by_age"].items():
                report += f"  - {age}: {round(rate, 2)}%\n"

    logging.info("Banking customer analysis report generated")

    # Write report to a file in the SQLite database directory
    try:
        sqlite_db_path = kwargs["params"]["sqlite_db_path"]
        report_dir = os.path.dirname(sqlite_db_path)
        report_path = os.path.join(report_dir, "banking_customer_analysis_report.txt")

        with open(report_path, "w") as f:
            f.write(report)

        logging.info(f"Analysis report saved to {report_path}")
    except Exception as e:
        logging.error(f"Error saving report to file: {str(e)}")

    # Log summary of the report
    logging.info(report[:500] + "... [truncated]")

    return "Banking customer analysis and reporting complete"


# Define task dependencies

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
