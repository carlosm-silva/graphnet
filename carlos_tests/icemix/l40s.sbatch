#!/bin/bash
#SBATCH -JDeepIceL40s				# Job name 
#SBATCH -Agts-itaboada3				# Charge account
#SBATCH -N1 					     # Number of nodes
#SBATCH --cpus-per-task=32              # Number of CPUs per task
#SBATCH --gres=gpu:l40s:8			# Number of nodes and GPUs required
#SBATCH -qembers					# QOS Name
#SBATCH --mem=0					# Memory per gpu (0 means all available)
#SBATCH --time=8:00:00				# Duration of the job (Ex: 120 mins)
#SBATCH -oReport-%j.out				# Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL		# Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu	# e-mail address for notifications

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix    # Change to your working directory 

# --- Load modules and activate conda environment -----------------------------
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1                                               # Load module dependencies
conda activate graphnet 

# --- Set environment variables ----------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0                                    # This is the default, set to 1 for debugging
export OMP_NUM_THREADS=4                                         # Bind CPU to GPUs
export MKL_NUM_THREADS=4                                         # Suggested by GPT, shouldn't really matter

# --- Create output directory ------------------------------------------------
export OUTPUT_DIR="/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix/telemetry_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# --- Print job information ---------------------------------------------------
echo "Logs → $OUTPUT_DIR"
echo "Job started at $(date)"
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
srun --cpu-bind=cores\
     torchrun --nproc_per_node=8 train_icemix.py --max-epochs 80 --early-stopping-patience 150 --batch-size 128 --num-workers 3  # Run test example
EXIT_CODE=$?

# ───────────────────────── Cleanup ───────────────────────────────────────────
echo "Training finished with code $EXIT_CODE - stopping monitors..."
kill $GPU_MON_PID $SYS_MON_PID 2>/dev/null

echo "All done at $(date)"
exit $EXIT_CODE
