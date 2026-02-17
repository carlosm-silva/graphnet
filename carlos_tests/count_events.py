import os

data_paths = {
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nue_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nue_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nue_database_part_1_validation_selection.csv",
    ],
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_numu_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_validation_selection.csv",
    ],
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nutau_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nutau_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nutau_database_part_1_validation_selection.csv",
    ],
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nue_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_validation_selection.csv",
    ],
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_numu_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_numu_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_numu_database_part_1_validation_selection.csv",
    ],
    "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nutau_database_part_1.db": [
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nutau_database_part_1_training_selection.csv",
        "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nutau_database_part_1_validation_selection.csv",
    ],
}

total_training = 0
total_validation = 0

print("Checking files...")
for db_path, file_list in data_paths.items():
    print(f"Processing DB: {os.path.basename(db_path)}")
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"  ERROR: File not found: {file_path}")
            continue

        try:
            with open(file_path, "r") as f:
                # Count lines and subtract header
                lines = sum(1 for line in f)
                count = max(0, lines - 1)

            fname = os.path.basename(file_path)

            if "training_selection" in fname:
                total_training += count
                print(f"  Training: {count} events ({fname})")
            elif "validation_selection" in fname:
                total_validation += count
                print(f"  Validation: {count} events ({fname})")
            else:
                print(f"  UNKNOWN TYPE: {count} events ({fname})")

        except Exception as e:
            print(f"  Error reading file {file_path}: {e}")

print("\nSummary:")
print("-" * 20)
print("[Before augmentation]")
print(
    f"Total Training Events: {total_training} ({total_training/(total_training+total_validation)*100:.2f}%)"
)
print(
    f"Total Validation Events: {total_validation} ({total_validation/(total_training+total_validation)*100:.2f}%)"
)
print(f"Total Events: {total_training + total_validation}")
# Augmentation
total_training *= 6
print("[After augmentation]")
print(
    f"Total Training Events: {total_training} ({total_training/(total_training+total_validation)*100:.2f}%)"
)
print(
    f"Total Validation Events: {total_validation} ({total_validation/(total_training+total_validation)*100:.2f}%)"
)
print(f"Total Events: {total_training + total_validation}")
