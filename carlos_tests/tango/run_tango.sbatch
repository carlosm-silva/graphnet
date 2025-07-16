#!/bin/bash
#SBATCH -JFull                                      # Job name 
#SBATCH -Agts-itaboada3                             # Charge account
#SBATCH -N1 
#SBATCH --cpus-per-task=8
#PBS -l file=100gb
#SBATCH --gres=gpu:H200:1                           # Number of nodes and GPUs required
#SBATCH -qinferno                                   # QOS Name
#SBATCH --mem-per-gpu=128G                          # Memory per gpu
#SBATCH --time=24:00:00                             # Duration of the job (Ex: 120 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu              # e-mail address for notifications


cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/tango                 

# --- Copy data to local NVMe for faster access ------------------------------
echo "Copying database files to local NVMe storage..."

# Copy database files to local NVMe (much faster than network storage)
cp "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_numu_database_part_1 (1).db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_nue_database_part_1 (1).db" "${TMPDIR}/" &
wait  # Wait for both copies to complete

echo "Data copy completed. Using local data at ${TMPDIR}"
export LOCAL_DATA_DIR="${TMPDIR}"  # Make available to training script


SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1                    # Load module dependencies
conda activate graphnet 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True                                                                     
srun python FullDataset.py  --max-epochs 36 --early-stopping-patience 150 --batch-size 100 --num-workers 8