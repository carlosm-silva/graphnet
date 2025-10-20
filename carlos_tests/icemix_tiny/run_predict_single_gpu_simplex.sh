#!/bin/bash

# Single-GPU prediction script for RTX6000 nodes - Simplex Version
# Run this from within your salloc'd interactive job

echo "=== DeepIce Single-GPU Simplex Prediction Script for RTX6000 ==="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPUs available:"
nvidia-smi --list-gpus

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny

# --- Create output directory ------------------------------------------------
export OUTPUT_DIR="/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/logs/prediction_single_simplex_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Outputs will be saved to: $OUTPUT_DIR"

# --- Copy data to local NVMe for faster access (if available) ---------------
if [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ]; then
    echo "Copying database files to local NVMe storage..."
    cp "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_numu_database_part_1 (1).db" "${TMPDIR}/" &
    cp "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_nue_database_part_1 (1).db" "${TMPDIR}/" &
    wait
    echo "Data copy completed. Using local data at ${TMPDIR}"
    export LOCAL_DATA_DIR="${TMPDIR}"
else
    echo "TMPDIR not available or not writable, using network storage directly"
    unset LOCAL_DATA_DIR
fi

# --- Load modules and activate conda environment -----------------------------
echo "Loading modules and activating conda environment..."
module load anaconda3/2022.05.0.1
conda activate graphnet 

# --- Set environment variables for single RTX6000 GPU ----------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=12                                        # Use more threads for single GPU
export MKL_NUM_THREADS=12

# RTX6000 optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_DEVICE_MAX_CONNECTIONS=32                           # Higher setting for single GPU

# --- Monitor GPU usage during prediction ------------------------------------
echo "Starting GPU monitor..."
nvidia-smi dmon -s pucvmet -d 5 -o TD -f "${OUTPUT_DIR}/nvidia_dmon_prediction_single_simplex.log" &
GPU_MON_PID=$!

# --- Run prediction on single GPU with simplex model -----------------------
echo "Starting prediction on single GPU (GPU 0) with simplex model..."
echo "Using larger batch size since we're using only one GPU..."

# Use regular python instead of torchrun to avoid distributed training issues
python train_icemix_mixed_tiny_simplex.py \
       --mode predict \
       --batch-size 512 \
       --num-workers 12 \
       --pin-memory \
       --persistent-workers \
       --gpus 0

EXIT_CODE=$?

# --- Cleanup -----------------------------------------------------------------
echo "Prediction finished with exit code: $EXIT_CODE"
echo "Stopping GPU monitor..."
kill $GPU_MON_PID 2>/dev/null

# Show prediction results location
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Simplex prediction completed successfully!"
    echo "Results should be available in:"
    echo "  /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results_simplex/"
else
    echo "❌ Simplex prediction failed with exit code: $EXIT_CODE"
    echo "Check logs in: $OUTPUT_DIR"
fi

echo "Finished at: $(date)"
exit $EXIT_CODE 