#!/bin/bash
#SBATCH -JDeepIceL40sMixedTiny 		    # Job name 
#SBATCH -Agts-itaboada3				    # Charge account
#SBATCH -N1 					        # Number of nodes
#SBATCH --cpus-per-task=32              # Number of CPUs per task (4 per GPU × 8 GPUs = 32 total)
#SBATCH --gres=gpu:l40s:8			    # Number of nodes and GPUs required
#SBATCH -qembers					    # QOS Name
#SBATCH --mem=0					        # Memory per gpu (0 means all available)
#SBATCH --time=8:00:00				    # Duration of the job (Ex: 120 mins)
#SBATCH -oReport-%j.out				    # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL	    # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu	# e-mail address for notifications
#SBATCH --exclude=atl1-1-03-004-31-0    # Exclude faulty node

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny    # Change to your working directory 

# --- Create output directory ------------------------------------------------
export OUTPUT_DIR="/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/logs/telemetry_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# --- Redirect output and error ----------------------------------------------
exec > "${OUTPUT_DIR}/Report-${SLURM_JOB_ID}.out" 2>&1

# --- Clean up Report file stubs ----------------------------------------------
rm -f Report-${SLURM_JOB_ID}.out

# --- Copy data to local NVMe for faster access ------------------------------
echo "Copying database files to local NVMe storage..."

# Copy database files to local NVMe (much faster than network storage)
cp "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/tango_data/my_numu_database_part_1 (1).db" "${TMPDIR}/" &
cp "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/tango_data/my_nue_database_part_1 (1).db" "${TMPDIR}/" &
wait  # Wait for both copies to complete

echo "Data copy completed. Using local data at ${TMPDIR}"
export LOCAL_DATA_DIR="${TMPDIR}"  # Make available to training script

# --- Load modules and activate conda environment -----------------------------
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1                                               # Load module dependencies
conda activate graphnet 

# --- Set environment variables ----------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0                                    # This is the default, set to 1 for debugging
export OMP_NUM_THREADS=4                                         # Bind CPU to GPUs
export MKL_NUM_THREADS=4                                         # Suggested by GPT, shouldn't really matter

# L40S specific optimizations
export TORCH_CUDNN_V8_API_ENABLED=1                             # Enable cuDNN v8 API for L40S
export NCCL_P2P_DISABLE=0                                       # Enable P2P for multi-GPU communication
export NCCL_IB_DISABLE=0                                        # Enable InfiniBand for fast communication
export CUDA_DEVICE_MAX_CONNECTIONS=32                           # Increase max connections for L40S

# --- Print job information ---------------------------------------------------
echo "Logs → $OUTPUT_DIR"
echo "Job started at $(date) on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Job queue: $SLURM_JOB_QUEUE"
echo "Job partition: $SLURM_JOB_PARTITION"
echo "Job nodes: $SLURM_JOB_NODELIST"

# --- monitor GPU every second ----------------------------------------------
echo "Starting GPU monitor..."
nvidia-smi dmon -s pucvmet -d 1 -o TD -f "${OUTPUT_DIR}/nvidia_dmon.log" &
GPU_MON_PID=$!

# --- monitor system resources every 2 s -------------------------------------
echo "Starting system monitor..."
python - <<'PY' &
import psutil, time, json, os
f = open(os.environ['OUTPUT_DIR'] + '/psutil_host.jsonl','a', buffering=1)
while True:
    rec = {
        "ts": time.time(),
        "cpu_pct": psutil.cpu_percent(None),
        "ram_used_mb": psutil.virtual_memory().used/2**20,
        "ram_pct": psutil.virtual_memory().percent,
        "disk_read_mb": psutil.disk_io_counters().read_bytes/2**20,
        "disk_write_mb": psutil.disk_io_counters().write_bytes/2**20,
    }
    f.write(json.dumps(rec)+'\n')
    time.sleep(2)
PY
SYS_MON_PID=$!

# --- Run the job -------------------------------------------------------------
echo "Training started..."
srun --cpu-bind=cores \
     torchrun --nproc_per_node=8 \
     train_icemix_mixed_tiny.py \
     --max-epochs 8 \
     --early-stopping-patience 150 \
     --batch-size 1024 \
     --num-workers 16 \
     --pin-memory \
     --persistent-workers
     # --accumulate-grad-batches 2     # Uncomment for effective batch size 4096
EXIT_CODE=$?

# ───────────────────────── Cleanup ───────────────────────────────────────────
echo "Training finished with code $EXIT_CODE - stopping monitors..."
kill $GPU_MON_PID $SYS_MON_PID 2>/dev/null

# No cleanup needed - ${TMPDIR} is automatically cleaned up when job completes

echo "All done at $(date)"
exit $EXIT_CODE
