# ingestion_task.py
import pandas as pd
import sqlite3


def ingest_data(raw_csv_file, golden_database, table_name="data_table"):
    """
    Ingest data from a CSV file into a SQLite database.

    Args:
        raw_csv_file (str): Path to the CSV file as Raw data.
        golden_database (str): Path to the SQLite database file in Golden data.
    """
    # Load data from CSV
    df = pd.read_csv(raw_csv_file)

    # Connect to SQLite database
    conn = sqlite3.connect(golden_database)

    # Write the DataFrame to the SQLite database
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Close the connection
    conn.close()

    print(f"Data ingested into {golden_database} successfully.")


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    import os

    args = sys.argv
    # Check if the correct number of arguments is provided
    if len(args) < 2:
        print("Usage: python ingestion.py <raw_csv_file> <golden_database>")
        sys.exit(1)

    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Ingest raw data from a CSV file into a golden table in the SQLite database."
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input CSV file (raw data)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output SQLite database file (golden database).",
    )
    parser.add_argument(
        "--table",
        default="data_table",
        help="Name of the table in the SQLite database (default: data_table).",
    )
    args = parser.parse_args()

    csv_file_path = args.input
    golden_db = args.output
    table_name = args.table

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"CSV file {csv_file_path} does not exist.")
        sys.exit(1)

    # Ingest data
    ingest_data(csv_file_path, golden_db, table_name)
