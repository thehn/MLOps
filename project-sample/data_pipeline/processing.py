# processing data in SQLite database

import sqlite3
import pandas as pd


# validate that table contains data
def validate_table(database_file, table_name):
    """
    Validate if the table in the SQLite database contains data.

    Args:
                database_file (str): Path to the SQLite database file.
                table_name (str): Name of the table to validate.
    """
    conn = sqlite3.connect(database_file)
    query = f"SELECT COUNT(*) FROM {table_name}"
    count = pd.read_sql_query(query, conn).iloc[0, 0]
    print(f"Number of records in {table_name}: {count}")
    if count == 0:
        print(f"Table {table_name} is empty.")
        conn.close()
        return False

    sample_query = f"SELECT * FROM {table_name} LIMIT 5"
    sample_data = pd.read_sql_query(sample_query, conn)
    print(f"Sample data from {table_name}:\n{sample_data}")

    conn.close()
    return count > 0

    # Example usage


if __name__ == "__main__":
    import sys
    import argparse

    args = sys.argv
    # Check if the correct number of arguments is provided
    if len(args) < 2:
        print("Usage: python processing.py <database_file>")
        sys.exit(1)

    # Get command line arguments
    parser = argparse.ArgumentParser(description="Process data from a SQLite database.")
    parser.add_argument(
        "--database", required=True, help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--table",
        default="data_table",
        help="Name of the table in the SQLite database (default: data_table).",
    )
    args = parser.parse_args()

    database_file = args.database
    table_name = args.table
    # Validate the table
    if not validate_table(database_file, table_name):
        print(f"Table {table_name} is empty or does not exist.")
        sys.exit(1)
