"""
Sample script to load data from the database and try basic data augmentation.
"""

import sqlite3
import pandas as pd
import os

data_paths = []

for pre in ["", "nugen_"]:
    for fla in ["e","mu","tau"]:
        data_paths.append(f"/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_{pre}nu{fla}_database_part_1.db")


for path in data_paths[-1:]:
    with sqlite3.connect(path) as conn:
        # Print database info
        print("\n" + "=" * 80)
        print(f"Database: {path}")
        print("=" * 80)
        
        # List all tables
        tables_cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in tables_cursor.fetchall()]
        print(f"\nTables in database: {tables}")
        
        # ========== TRUTH TABLE ==========
        print("\n" + "=" * 80)
        print("TRUTH TABLE")
        print("=" * 80)
        
        # Count total rows
        count_cursor = conn.execute("SELECT COUNT(*) FROM truth")
        total_rows = count_cursor.fetchone()[0]
        print(f"\nTotal rows: {total_rows:,}")
        
        # Get column names from a sample query
        cursor = conn.execute("SELECT * FROM truth LIMIT 1")
        column_names = [description[0] for description in cursor.description]
        sample_data = cursor.fetchone()
        
        # Print column names
        print(f"\nColumns ({len(column_names)}):")
        print("-" * 80)
        for i, col_name in enumerate(column_names):
            print(f"  {i+1:2d}. {col_name}")
        
        # Print data sample
        print(f"\nSample row (first of {total_rows:,}):")
        print("-" * 80)
        for col_name, value in zip(column_names, sample_data):
            # Format the value nicely
            if isinstance(value, float):
                formatted_value = f"{value:,.6f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            print(f"  {col_name:.<30} {formatted_value}")
        print("-" * 80)
        

        # ========== SRTInIcePulses TABLE ==========
        print("\n" + "=" * 80)
        print("SRTInIcePulses TABLE")
        print("=" * 80)
        
        # Count total rows
        count_cursor_pulses = conn.execute("SELECT COUNT(*) FROM SRTInIcePulses")
        total_rows_pulses = count_cursor_pulses.fetchone()[0]
        print(f"\nTotal rows: {total_rows_pulses:,}")
        
        # Get column names from a sample query
        cursor_pulses = conn.execute("SELECT * FROM SRTInIcePulses LIMIT 1")
        column_names_pulses = [description[0] for description in cursor_pulses.description]
        sample_data_pulses = cursor_pulses.fetchone()
        
        # Print column names
        print(f"\nColumns ({len(column_names_pulses)}):")
        print("-" * 80)
        for i, col_name in enumerate(column_names_pulses):
            print(f"  {i+1:2d}. {col_name}")
        
        # Print data sample
        print(f"\nSample row (first of {total_rows_pulses:,}):")
        print("-" * 80)
        for col_name, value in zip(column_names_pulses, sample_data_pulses):
            # Format the value nicely
            if isinstance(value, float):
                formatted_value = f"{value:,.6f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            print(f"  {col_name:.<30} {formatted_value}")
        print("-" * 80)




# NumuValidation = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_validation_selection.csv"
# numu_validation_selections = pd.read_csv(NumuValidation)
# print(numu_validation_selections.head())

# # Print all files in the directory
# directory = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data"
# files = os.listdir(directory)
# # Filter only files that start with P14
# files = [file for file in files if file.startswith("P14")]
# # must contain at least one of the following strings: nue, numu, nutau
# files = [file for file in files if any(string in file for string in ["nue", "numu", "nutau"])]
# for i, file in enumerate(files):
#     df = pd.read_csv(directory + "/" + file)
#     # get the max value of the first column
#     print(f"{i+1}: {file} {df.max()}")