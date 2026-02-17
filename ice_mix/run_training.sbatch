#!/bin/bash
#SBATCH -JIceMixTraining           	    # Job name 
#SBATCH -Agts-pli77-ideas_l40s			# Charge account
#SBATCH -N1 					        # Number of nodes
#SBATCH --cpus-per-task=32              # Number of CPUs per task (4 per GPU × 8 GPUs = 32 total)
#SBATCH --gres=gpu:l40s:8			# Number of nodes and GPUs required
#SBATCH -qinferno					    # QOS Name
#SBATCH --mem=0					        # Memory per gpu (0 means all available)
#SBATCH --time=72:00:00				    # Duration of the job (8 hours for full training)
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL	    # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu	# e-mail address for notifications
#SBATCH --nodelist=atl1-1-01-010-29-0,atl1-1-01-010-31-0,atl1-1-01-010-33-0


# --- Set working directory ---------------------------------------------------
cd /storage/project/r-itaboada3-0/cfilho3/graphnet

# --- Create output directory for logs if it doesn't exist --------------------
mkdir -p ice_mix/logs/sbatch_reports

# --- Copy data to local NVMe for faster access ------------------------------
echo "Copying database files to local NVMe storage..."

# Check if TMPDIR is set, otherwise use a default or fail
if [ -z "$TMPDIR" ]; then
    echo "TMPDIR is not defined. Using /tmp"
    export TMPDIR="/tmp"
fi

# Copy database files to local NVMe
# Note: Using the standard databases as default. 
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/my_numu_database_part_1 (1).db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/my_nue_database_part_1 (1).db" "${TMPDIR}/" &
wait

echo "Data copy completed. Using local data at ${TMPDIR}"
export LOCAL_DATA_DIR="${TMPDIR}"

# --- Load modules and activate conda environment -----------------------------
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1
conda activate graphnet

# Load environment variables from .env
if [ -f ice_mix/.env ]; then
    export $(grep -v '^#' ice_mix/.env | xargs)
fi

# --- Set environment variables ----------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# L40S specific optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=32

# --- Run the job -------------------------------------------------------------
echo "Job started at $(date) on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Run with python directly. Hydra will handle loading config.yaml
srun --cpu-bind=cores \
     torchrun --nproc_per_node=8 \
     ice_mix/train.py

EXIT_CODE=$?
echo "Job finished with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
