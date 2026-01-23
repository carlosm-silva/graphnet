#!/bin/bash
#SBATCH -JAugmentData  # Job name 
#SBATCH -Agts-itaboada3				    # Charge account
#SBATCH -N1 					        # Number of nodes
#SBATCH --cpus-per-task=128             # Number of CPUs per task
#SBATCH -qembers					    # QOS Name
#SBATCH --constraint="cpu-amd"          # Dual AMD Epyc 9534 CPUs @ 2.45 GHz (128 cores/node)
#SBATCH --mem=0                         # Request all available memory
#SBATCH --time=1:00:00				    # Duration of the job
#SBATCH -oReport-%j.out				    # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL	    # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu	# e-mail address for notifications


# --- Copy data to local NVMe for faster access ------------------------------
echo "Copying database files to local NVMe storage..."

# Copy database files to local NVMe (much faster than network storage)
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nue_database_part_1.db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_numu_database_part_1.db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nutau_database_part_1.db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nue_database_part_1.db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_numu_database_part_1.db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/P14_nugen_nutau_database_part_1.db" "${TMPDIR}/" &
wait  # Wait for all copies to complete

echo "Data copy completed. Using local data at ${TMPDIR}"
export LOCAL_DATA_DIR="${TMPDIR}"  # Make available to script

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests   # Change to your working directory 

module load anaconda3/2022.05.0.1                                               # Load module dependencies
conda activate graphnet 

srun python data_augmentation.py