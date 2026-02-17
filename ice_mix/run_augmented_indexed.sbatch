#!/bin/bash
#SBATCH -JIceMixAugmented-Rotation                         # Job name 
#SBATCH -Agts-pli77-ideas_l40s                       # Charge account
#SBATCH -N1                                          # Number of nodes
#SBATCH --cpus-per-task=32                           # Number of CPUs per task
#SBATCH --gres=gpu:l40s:8                            # GPUs required
#SBATCH -qinferno                                    # QOS Name
#SBATCH --mem=0                                      # Memory per gpu (0 means all available)
#SBATCH --time=72:00:00                              # Duration
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-Augmented-Rotation-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL                   # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu               # Notifications
#SBATCH --nodelist=atl1-1-03-004-31-0,atl1-1-01-010-29-0,atl1-1-01-010-31-0,atl1-1-03-004-29-0,atl1-1-03-007-29-0,atl1-1-03-007-31-0

# --- Set working directory ---------------------------------------------------
cd /storage/project/r-itaboada3-0/cfilho3/graphnet
mkdir -p ice_mix/logs/sbatch_reports

# --- Copy AUGMENTED data to local NVMe --------------------------------------
echo "Copying AUGMENTED database files to local NVMe..."
if [ -z "$TMPDIR" ]; then export TMPDIR="/tmp"; fi

# Ensure we found the correct augmented files from scratch earlier
cp "/storage/scratch1/8/cfilho3/P14_numu_database_part_1_augmented.db" "${TMPDIR}/" &
cp "/storage/scratch1/8/cfilho3/P14_nue_database_part_1_augmented.db" "${TMPDIR}/" &
wait

echo "Augmented data copy completed. Using local data at ${TMPDIR}"
export LOCAL_DATA_DIR="${TMPDIR}"

# --- Environment -------------------------------------------------------------
module load anaconda3/2022.05.0.1
conda activate graphnet

# Load environment variables from .env
if [ -f ice_mix/.env ]; then
    export $(grep -v '^#' ice_mix/.env | xargs)
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=32

# --- Run Augmented Training -------------------------------------------------
# Override data config to use 'augmented_rotation'
echo "Job started at $(date) on $SLURM_JOB_NODELIST"
srun --cpu-bind=cores \
     torchrun --nproc_per_node=8 \
     ice_mix/train.py \
     data=augmented_rotation \
     project_name="IceMix-Augmented-Rotation"

exit $?
