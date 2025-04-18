# feature_engineering_task.py: Performs feature engineering on the data.
# Input: SQLite database file
# Output: SQLite database file with new features added.


def engineering_features(
    golden_database, golden_table_name, feature_store_database, feature_store_table_name
):
    """
    Perform feature engineering on the data in the SQLite database.

    Args:
        golden_database (str): Path to the SQLite database file containing the raw data.
        golden_table_name (str): Name of the table in the SQLite database.
        feature_store_database (str): Path to the SQLite database file for storing features.
        feature_store_table_name (str): Name of the table in the SQLite database for storing features.

    """
    import sqlite3
    import pandas as pd

    # Connect to SQLite database
    golden_conn = sqlite3.connect(golden_database)

    # Read data from the SQLite database into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {golden_table_name}", golden_conn)

    # Perform feature engineering tasks here

    # Handle missing values
    df = df.dropna()

    dropped_columns = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=dropped_columns)

    # Convert categorical variables to numeric

    # Convert Gender
    gender_mapping = {"Female": 0, "Male": 1}
    df["Gender"] = df["Gender"].map(gender_mapping)

    # Convert Geography
    geo_mapping = {"France": 0, "Spain": 1, "Germany": 2}
    df["Geography"] = df["Geography"].map(geo_mapping)

    # Others
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes

    # Write the DataFrame to the SQLite database
    feature_store_conn = sqlite3.connect(feature_store_database)
    df.to_sql(
        feature_store_table_name, feature_store_conn, if_exists="replace", index=False
    )

    # Close the connections
    golden_conn.close()
    feature_store_conn.close()

    print(
        f"Feature engineering completed and saved to {feature_store_database} successfully."
        f"Table name: {feature_store_table_name}"
    )


# Example usage
if __name__ == "__main__":
    import argparse
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Perform feature engineering on data in a SQLite database."
    )
    parser.add_argument(
        "--golden-db",
        required=True,
        help="Path to the input SQLite database file (golden database).",
    )
    parser.add_argument(
        "--feature-store-db",
        required=True,
        help="Path to the output SQLite database file (feature store database).",
    )
    parser.add_argument(
        "--golden-table",
        required=True,
        help="Name of the input golden table in the SQLite database.",
    )
    parser.add_argument(
        "--feature-store-table",
        required=True,
        help="Name of the output feature store table in the SQLite database.",
    )
    args = parser.parse_args()

    golden_db = args.golden_db
    fs_db = args.feature_store_db
    golden_tbl = args.golden_table
    fs_tbl = args.feature_store_table

    # Check if the input database file exists
    if not os.path.exists(golden_db):
        print(f"Input database file {golden_db} does not exist.")
        exit(1)

    # Perform feature engineering
    engineering_features(
        golden_db,
        golden_tbl,
        fs_db,
        fs_tbl,
    )
