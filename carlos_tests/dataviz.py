import pandas as pd
import sqlite3
from graphnet.data.constants import FEATURES, TRUTH

# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

db_path = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_numu_database_part_1 (1).db"

if __name__ == '__main__':
    # Import database and show head
    conn = sqlite3.connect(db_path)

    # List db contents
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    print("Tables in the database:")
    print(tables)

    # Load a specific table
    table_name = tables.iloc[0]['name']  # Change this to the desired table
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    print(f"\nData from table '{table_name}':")
    print(df.head())

    conn.close()
