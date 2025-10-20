#!/bin/bash
#SBATCH -J DeepIcePredictH200              # Job name
#SBATCH -A gts-itaboada3                   # Charge account
#SBATCH -N 1                               # Number of nodes
#SBATCH --cpus-per-task=8                  # Number of CPUs per task
#SBATCH --gres=gpu:h200:1                  # 1 H200 GPU
#SBATCH -q embers                          # QOS Name
#SBATCH --mem=64G                          # Memory in GB
#SBATCH --time=01:00:00                    # Duration of the job (1 hour)
#SBATCH -o predict_h200_%j.out             # Output file
#SBATCH -e predict_h200_%j.err             # Error file
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu     # e-mail address for notifications

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny

# --- Create output directory ------------------------------------------------
export OUTPUT_DIR="/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/logs/prediction_h200_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# --- Redirect output and error ----------------------------------------------
exec > "${OUTPUT_DIR}/predict_h200_${SLURM_JOB_ID}.out" 2>&1

# --- Print job information ---------------------------------------------------
echo "=== DeepIce H200 Prediction Job ==="
echo "Job started at $(date) on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Job queue: $SLURM_JOB_QUEUE"
echo "Job partition: $SLURM_JOB_PARTITION"
echo "Job nodes: $SLURM_JOB_NODELIST"
echo "Outputs will be saved to: $OUTPUT_DIR"

# --- Load modules and activate conda environment -----------------------------
echo "Loading modules and activating conda environment..."
module load anaconda3/2022.05.0.1
conda activate graphnet

# --- Set environment variables for H200 GPU --------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# H200 optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=32

# --- Monitor GPU usage during prediction ------------------------------------
echo "Starting GPU monitor..."
nvidia-smi dmon -s pucvmet -d 5 -o TD -f "${OUTPUT_DIR}/nvidia_dmon_prediction.log" &
GPU_MON_PID=$!

# --- Run the prediction script ----------------------------------------------
echo "Starting prediction script..."
bash /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/run_predict_single_gpu.sh

EXIT_CODE=$?

# --- Cleanup -----------------------------------------------------------------
echo "Prediction finished with exit code: $EXIT_CODE"
echo "Stopping GPU monitor..."
kill $GPU_MON_PID 2>/dev/null

# Show prediction results location
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Prediction completed successfully!"
    echo "Results should be available in:"
    echo "  /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/"
else
    echo "❌ Prediction failed with exit code: $EXIT_CODE"
    echo "Check logs in: $OUTPUT_DIR"
fi

echo "Finished at: $(date)"
exit $EXIT_CODE
