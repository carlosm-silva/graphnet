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

GOOD=$(
python - <<'PY'
import subprocess, torch, re
good=[]
n=torch.cuda.device_count()
for i in range(n):
    try:
        out=subprocess.check_output(["nvidia-smi","-i",str(i),"-q","-d","ECC"], text=True, stderr=subprocess.DEVNULL)
        m=re.search(r"Volatile Uncorr\. ECC.*?:\s+(\d+)", out)
        if m and m.group(1) not in ("0","N/A"):
            continue
    except Exception:
        pass
    try:
        torch.cuda.set_device(i)
        a=torch.randn(512,512, device=f"cuda:{i}")
        b=torch.randn(512,512, device=f"cuda:{i}")
        (a@b).sum().item(); torch.cuda.synchronize(i)
        good.append(str(i))
    except Exception:
        pass
print(",".join(good))
PY
)
if [ -z "$GOOD" ]; then echo "[FATAL] no healthy GPUs"; exit 42; fi
export CUDA_VISIBLE_DEVICES="${GOOD}"
echo "Using GPUs (visible): ${CUDA_VISIBLE_DEVICES}"

# Load environment variables from .env
if [ -f ice_mix/.env ]; then
    export $(grep -v '^#' ice_mix/.env | xargs)
fi

# Redirect WandB cache to Scratch
export WANDB_CACHE_DIR="/storage/scratch1/8/cfilho3/wandb_cache"
export WANDB_DIR="/storage/scratch1/8/cfilho3/wandb_dir"
export WANDB_DATA_DIR="/storage/scratch1/8/cfilho3/wandb_data"

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
