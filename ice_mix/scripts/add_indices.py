import sqlite3
import argparse
import sys
import os


def check_and_create_index(db_path, table_name, column_name):
    print(f"Checking database: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        if not cursor.fetchone():
            print(f"  Table '{table_name}' does not exist.")
            conn.close()
            return

        # Check existing indices
        cursor.execute(f"PRAGMA index_list({table_name})")
        indices = cursor.fetchall()

        index_exists = False
        for idx in indices:
            idx_name = idx[1]
            cursor.execute(f"PRAGMA index_info({idx_name})")
            cols = cursor.fetchall()
            # Check if this index is on the target column (single column index)
            if len(cols) == 1 and cols[0][2] == column_name:
                index_exists = True
                print(
                    f"  Index on '{table_name}({column_name})' already exists: {idx_name}"
                )
                break

        if not index_exists:
            print(f"  Creating index on '{table_name}({column_name})'...")
            index_name = f"{table_name}_{column_name}_index_auto"
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name})"
            )
            conn.commit()
            print(f"  Successfully created index: {index_name}")

        conn.close()

    except sqlite3.Error as e:
        print(f"  SQLite error: {e}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Add indices to GraphNeT SQLite databases."
    )
    parser.add_argument("databases", nargs="+", help="Path to SQLite database files")
    args = parser.parse_args()

    for db_path in args.databases:
        check_and_create_index(db_path, "truth", "event_no")
        check_and_create_index(db_path, "SRTInIcePulses", "event_no")


if __name__ == "__main__":
    main()
