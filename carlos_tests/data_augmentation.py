"""
Data augmentation script for neutrino physics databases.
Creates augmented database files by applying random rotations to training data.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
import concurrent.futures
from tqdm import tqdm

# Configuration
data_paths = {'/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nue_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nue_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nue_database_part_1_validation_selection.csv'],
 '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_numu_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_validation_selection.csv'],
 '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nutau_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nutau_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nutau_database_part_1_validation_selection.csv'],
 '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nue_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_validation_selection.csv'],
 '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_numu_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_numu_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_numu_database_part_1_validation_selection.csv'],
 '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nutau_database_part_1.db': ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nutau_database_part_1_training_selection.csv',
  '/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nutau_database_part_1_validation_selection.csv']}


# Check if running with local data (NVMe)
if "LOCAL_DATA_DIR" in os.environ:
    local_dir = os.environ["LOCAL_DATA_DIR"]
    print(f"Using local data directory: {local_dir}")
    
    # Update keys in data_paths to point to local directory
    new_data_paths = {}
    for db_path, csv_paths in data_paths.items():
        db_basename = os.path.basename(db_path)
        local_db_path = os.path.join(local_dir, db_basename)
        new_data_paths[local_db_path] = csv_paths
        
    data_paths = new_data_paths

scratch_directory = "/storage/scratch1/8/cfilho3"


def apply_rotation_2d(x, y, angle):
    """
    Apply 2D rotation to x, y coordinates.
    
    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
        angle: rotation angle in radians
    
    Returns:
        rotated_x, rotated_y
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotated_x = x * cos_angle - y * sin_angle
    rotated_y = x * sin_angle + y * cos_angle
    
    rotated_x = x * cos_angle - y * sin_angle
    rotated_y = x * sin_angle + y * cos_angle
    
    return rotated_x, rotated_y


def get_rotation_angles(event_no, num_rotations):
    """
    Generate deterministic rotation angles based on event_no.
    Ensures consistency between truth and pulse tables.
    """
    # Use event_no as seed. Ensure it's a valid uint32 seed.
    seed = abs(int(event_no)) % (2**32)
    state = np.random.RandomState(seed)
    return state.uniform(0, 2 * np.pi, num_rotations)


def augment_database(db_path, training_csv, validation_csv, output_path, num_rotations=5, db_name=""):
    """
    Create augmented database with rotated training data.
    
    Args:
        db_path: Path to source database
        training_csv: Path to training selection CSV
        validation_csv: Path to validation selection CSV
        output_path: Path to output augmented database
        num_rotations: Number of random rotations per training event (default: 5)
    """
    
    print(f"\nProcessing: {os.path.basename(db_path)}")
    print(f"Output: {os.path.basename(output_path)}")
    
    # Load training and validation selections
    print("Loading selection files...")
    training_df = pd.read_csv(training_csv)
    validation_df = pd.read_csv(validation_csv)
    
    training_events = set(training_df['event_no'].values)
    validation_events = set(validation_df['event_no'].values)
    
    print(f"Training events: {len(training_events)}")
    print(f"Validation events: {len(validation_events)}")
    
    # Connect to source and destination databases
    src_conn = sqlite3.connect(db_path)
    dst_conn = sqlite3.connect(output_path)
    
    # Create tables in destination database
    print("Creating tables in destination database...")
    create_tables(src_conn, dst_conn)
    
    # Process truth table
    print(f"[{db_name}] Processing truth table...")
    process_truth_table(src_conn, dst_conn, training_events, validation_events, num_rotations, db_name)
    
    # Process SRTInIcePulses table
    print(f"[{db_name}] Processing SRTInIcePulses table...")
    process_pulses_table(src_conn, dst_conn, training_events, validation_events, num_rotations, db_name)
    
    # Commit and close
    dst_conn.commit()
    src_conn.close()
    dst_conn.close()
    
    print(f"Completed: {os.path.basename(output_path)}")


def create_tables(src_conn, dst_conn):
    """Create tables in destination database with same schema as source."""
    
    # Get table schemas from source
    cursor = src_conn.cursor()
    
    # Get truth table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='truth'")
    truth_schema = cursor.fetchone()[0]
    
    # Get SRTInIcePulses table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='SRTInIcePulses'")
    pulses_schema = cursor.fetchone()[0]
    
    # Create tables in destination
    dst_cursor = dst_conn.cursor()
    dst_cursor.execute(truth_schema)
    dst_cursor.execute(pulses_schema)
    dst_conn.commit()


def process_truth_table(src_conn, dst_conn, training_events, validation_events, num_rotations, db_name=""):
    """Process and augment truth table in chunks."""
    
    # Get total count for progress bar
    cursor = src_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM truth")
    total_rows = cursor.fetchone()[0]
    
    chunk_size = 50000
    total_augmented = 0
    
    # Iterate in chunks
    for chunk in tqdm(pd.read_sql_query("SELECT * FROM truth", src_conn, chunksize=chunk_size), 
                      total=(total_rows // chunk_size) + 1, 
                      desc=f"{db_name[:15]:<15} Truth",
                      position=1, # Let workers attempt to not overlap too much, though in SBatch it's just log lines
                      leave=False):
        
        augmented_rows = []
        
        # Identify training/validation rows
        is_training = chunk['event_no'].isin(training_events)
        is_validation = chunk['event_no'].isin(validation_events)
        
        # Add validation events as-is
        val_rows = chunk[is_validation].copy()
        if not val_rows.empty:
            augmented_rows.append(val_rows)
        
        # Add original training events
        train_rows = chunk[is_training].copy()
        if not train_rows.empty:
            augmented_rows.append(train_rows)
            
            # Generate augmentations
            # We iterate by rotation index to process vectorially
            for rotation_id in range(1, num_rotations + 1):
                aug_chunk = train_rows.copy()
                
                # Get angles for each event
                # We map event_no to the specific angle for this rotation_id
                # (rotation_id is 1-based index, so we use rotation_id-1 for list access)
                angles = aug_chunk['event_no'].apply(
                    lambda x: get_rotation_angles(x, num_rotations)[rotation_id-1]
                )
                
                # Modify event_no
                aug_chunk['event_no'] = aug_chunk['event_no'] + rotation_id * 10**10
                
                # Apply rotation to position
                rotated_x, rotated_y = apply_rotation_2d(
                    aug_chunk['position_x'], 
                    aug_chunk['position_y'], 
                    angles
                )
                aug_chunk['position_x'] = rotated_x
                aug_chunk['position_y'] = rotated_y
                
                # Apply rotation to azimuth
                aug_chunk['azimuth'] = (aug_chunk['azimuth'] + angles) % (2 * np.pi)
                
                augmented_rows.append(aug_chunk)
        
        # Write chunk results to DB
        if augmented_rows:
            combined_df = pd.concat(augmented_rows, ignore_index=True)
            combined_df.to_sql('truth', dst_conn, if_exists='append', index=False)
            total_augmented += len(combined_df)

    print(f"  Original truth rows: {total_rows}")
    print(f"  Augmented truth rows: {total_augmented}")


def process_pulses_table(src_conn, dst_conn, training_events, validation_events, num_rotations, db_name=""):
    """Process and augment SRTInIcePulses table in chunks."""
    
    # Get total count for progress bar
    cursor = src_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM SRTInIcePulses")
    total_rows = cursor.fetchone()[0]
    
    chunk_size = 200000  # Adjust based on memory
    total_augmented = 0
    
    # Iterate in chunks
    for chunk in tqdm(pd.read_sql_query("SELECT * FROM SRTInIcePulses", src_conn, chunksize=chunk_size),
                      total=(total_rows // chunk_size) + 1,
                      desc=f"{db_name[:15]:<15} Pulses",
                      position=1,
                      leave=False):
        
        augmented_dfs = []
        
        # Identify training/validation rows
        is_training = chunk['event_no'].isin(training_events)
        is_validation = chunk['event_no'].isin(validation_events)
        
        # Add validation events as-is
        val_rows = chunk[is_validation].copy()
        if not val_rows.empty:
            augmented_dfs.append(val_rows)
        
        # Add original training events
        train_rows = chunk[is_training].copy()
        if not train_rows.empty:
            augmented_dfs.append(train_rows)
            
            # Identify unique training events in this chunk for angle lookup
            unique_train_events = train_rows['event_no'].unique()
            
            # Generate augmented data for each rotation
            for rotation_id in range(1, num_rotations + 1):
                
                # Pre-calculate angles for relevant events to speed up logical step
                # Map: event_no -> angle
                # We use the deterministic helper
                angle_map = {
                    evt: get_rotation_angles(evt, num_rotations)[rotation_id-1]
                    for evt in unique_train_events
                }
                
                aug_chunk = train_rows.copy()
                
                # Map angles to rows
                row_angles = aug_chunk['event_no'].map(angle_map)
                
                # Modify event_no
                aug_chunk['event_no'] = aug_chunk['event_no'] + rotation_id * 10**10
                
                # Apply rotation to DOM position (vectorized)
                rotated_x, rotated_y = apply_rotation_2d(
                    aug_chunk['dom_x'], 
                    aug_chunk['dom_y'], 
                    row_angles
                )
                aug_chunk['dom_x'] = rotated_x
                aug_chunk['dom_y'] = rotated_y
                
                augmented_dfs.append(aug_chunk)
        
        # Write chunk results to DB
        if augmented_dfs:
            combined_df = pd.concat(augmented_dfs, ignore_index=True)
            combined_df.to_sql('SRTInIcePulses', dst_conn, if_exists='append', index=False)
            total_augmented += len(combined_df)
    
    print(f"  Original pulses rows: {total_rows}")
    print(f"  Augmented pulses rows: {total_augmented}")


def main():
    """Main execution function."""
    
    # Create scratch directory if it doesn't exist
    os.makedirs(scratch_directory, exist_ok=True)
    
    print("="*80)
    print("Data Augmentation Script")
    print("="*80)
    print(f"Scratch directory: {scratch_directory}")
    print(f"Number of databases to process: {len(data_paths)}")
    print(f"Rotations per training event: 5")
    print("="*80)
    

def process_single_database(db_path, selection_files, scratch_directory):
    """Process a single database. Used for parallel execution."""
    
    training_csv = selection_files[0]
    validation_csv = selection_files[1]
    
    # Create output filename
    db_basename = os.path.basename(db_path)
    output_filename = db_basename.replace('.db', '_augmented.db')
    output_path = os.path.join(scratch_directory, output_filename)
    
    # Check if input files exist
    if not os.path.exists(db_path):
        return f"\nWARNING: Database not found: {db_path}"
    
    if not os.path.exists(training_csv):
        return f"\nWARNING: Training CSV not found: {training_csv}"
    
    if not os.path.exists(validation_csv):
        return f"\nWARNING: Validation CSV not found: {validation_csv}"
        
    # Checkpoint: Check if output already exists
    if os.path.exists(output_path):
        return f"\nSKIPPING: {output_filename} already exists."
    
    # Determine working path (NVMe if available, else Scratch)
    if "LOCAL_DATA_DIR" in os.environ:
        # We work in local dir, then copy to scratch
        working_path = os.path.join(os.environ["LOCAL_DATA_DIR"], output_filename)
        # Ensure directory exists (though TMPDIR should exist)
        os.makedirs(os.path.dirname(working_path), exist_ok=True)
    else:
        # We work directly in scratch
        working_path = output_path
        
    # Use temporary file logic on the working path
    temp_working_path = working_path + ".partial"
    
    # Process database
    try:
        # Clean up potential leftover partial file
        if os.path.exists(temp_working_path):
            os.remove(temp_working_path)
            
        augment_database(db_path, training_csv, validation_csv, temp_working_path, db_name=db_basename)
        
        # Atomic finalize
        if "LOCAL_DATA_DIR" in os.environ:
            # 1. Rename to final name on NVMe
            if os.path.exists(working_path):
                os.remove(working_path)
            os.rename(temp_working_path, working_path)
            
            # 2. Copy from NVMe to Scratch (Cross-device copy)
            shutil.copy2(working_path, output_path)
            
            # 3. Cleanup local NVMe copy to save space
            os.remove(working_path)
        else:
            # Simple rename on same filesystem
            os.rename(temp_working_path, output_path)
            
        return f"Completed: {output_filename}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up partial file on failure
        if os.path.exists(temp_working_path):
            os.remove(temp_working_path)
        return f"\nERROR processing {db_basename}: {str(e)}"


def main():
    """Main execution function."""
    
    # Create scratch directory if it doesn't exist
    os.makedirs(scratch_directory, exist_ok=True)
    
    print("="*80)
    print("Data Augmentation Script")
    print("="*80)
    print(f"Scratch directory: {scratch_directory}")
    print(f"Number of databases to process: {len(data_paths)}")
    print(f"Rotations per training event: 5")
    
    # Check if passing LOCAL_DATA_DIR
    if "LOCAL_DATA_DIR" in os.environ:
        print(f"Using parallel processing with local NVMe storage: {os.environ['LOCAL_DATA_DIR']}")
    else:
        print("Using parallel processing on scratch storage")
        
    print("="*80)
    
    # Use ProcessPoolExecutor for parallel processing
    # We use max_workers=None which defaults to number of processors on the machine (128 on this node)
    # Since we have only 6 files, it will spawn 6 processes.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for db_path, selection_files in data_paths.items():
            futures.append(
                executor.submit(process_single_database, db_path, selection_files, scratch_directory)
            )
        
        # Monitor completion
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing databases"):
            try:
                result = future.result()
                # print(result) # Silence individual file output to keep tqdm clean, or print below
                tqdm.write(result) # Use tqdm.write to print safely without breaking the bar
            except Exception as e:
                tqdm.write(f"Task generated an exception: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*80)
    print("Data augmentation complete!")
    print("="*80)


if __name__ == "__main__":
    main()