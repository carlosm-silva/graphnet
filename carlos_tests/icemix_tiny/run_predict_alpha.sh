#!/bin/bash

# Parallel single-GPU prediction script for RTX6000 nodes - Alpha Experiments
# Run this from within your salloc'd interactive job

echo "=== DeepIce Parallel Alpha Prediction Script for RTX6000 ==="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "GPUs available:"
nvidia-smi --list-gpus

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny

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

# Function to run prediction for a specific alpha on a specific GPU
run_prediction_parallel() {
    local alpha=$1
    local gpu_id=$2
    local alpha_str=$(echo $alpha | sed 's/\./_/g')
    
    echo "Starting prediction for alpha = $alpha on GPU $gpu_id..."
    
    # Create alpha-specific output directory
    local OUTPUT_DIR="/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/logs/prediction_alpha_${alpha_str}_gpu${gpu_id}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    # Monitor GPU usage during prediction
    nvidia-smi dmon -s pucvmet -d 5 -o TD -f "${OUTPUT_DIR}/nvidia_dmon_prediction_alpha_${alpha_str}_gpu${gpu_id}.log" &
    local GPU_MON_PID=$!
    
    # Run prediction with alpha-specific parameters on specific GPU
    echo "Running alpha = $alpha on GPU $gpu_id with larger batch size..."
    
    # Use regular python instead of torchrun to avoid distributed training issues
    # Capture both stdout and stderr to separate log files for debugging
    python train_icemix_mixed_tiny_alpha.py \
           --mode predict \
           --batch-size 256 \
           --num-workers 12 \
           --pin-memory \
           --persistent-workers \
           --alpha $alpha \
           --gpus $gpu_id \
           > "${OUTPUT_DIR}/python_stdout_alpha_${alpha_str}_gpu${gpu_id}.log" \
           2> "${OUTPUT_DIR}/python_stderr_alpha_${alpha_str}_gpu${gpu_id}.log"
    
    local EXIT_CODE=$?
    
    # Cleanup for this alpha
    echo "Prediction for alpha = $alpha on GPU $gpu_id finished with exit code: $EXIT_CODE"
    kill $GPU_MON_PID 2>/dev/null
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Prediction completed successfully for alpha = $alpha on GPU $gpu_id!"
        echo "Results saved in: /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/"
        echo "Logs saved in: $OUTPUT_DIR"
    else
        echo "❌ Prediction failed for alpha = $alpha on GPU $gpu_id with exit code: $EXIT_CODE"
        echo "Check logs in: $OUTPUT_DIR"
        echo "  - GPU monitoring: nvidia_dmon_prediction_alpha_${alpha_str}_gpu${gpu_id}.log"
        echo "  - Python stdout: python_stdout_alpha_${alpha_str}_gpu${gpu_id}.log"
        echo "  - Python stderr: python_stderr_alpha_${alpha_str}_gpu${gpu_id}.log"
    fi
    
    return $EXIT_CODE
}

# Run predictions in parallel on different GPUs
echo "Running predictions for all alpha experiments in parallel..."
echo "Alpha 0.026 on GPU 0, Alpha 0.040 on GPU 1, Alpha 0.060 on GPU 2"

# Start all three predictions in parallel
run_prediction_parallel 0.026 0 &
PID_026=$!

run_prediction_parallel 0.040 1 &
PID_040=$!

run_prediction_parallel 0.060 2 &
PID_060=$!

# Wait for all predictions to complete
echo "Waiting for all predictions to complete..."
wait $PID_026
EXIT_026=$?

wait $PID_040
EXIT_040=$?

wait $PID_060
EXIT_060=$?

# Summary
echo ""
echo "=== Parallel Prediction Summary ==="
echo "Alpha 0.026 (GPU 0): $([ $EXIT_026 -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo "Alpha 0.040 (GPU 1): $([ $EXIT_040 -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
echo "Alpha 0.060 (GPU 2): $([ $EXIT_060 -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"

if [ $EXIT_026 -eq 0 ] && [ $EXIT_040 -eq 0 ] && [ $EXIT_060 -eq 0 ]; then
    echo ""
    echo "🎉 All alpha predictions completed successfully in parallel!"
    echo "Results are available in:"
    echo "  /storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/results/"
    echo ""
    echo "Each alpha experiment will have its own subdirectory with the results."
else
    echo ""
    echo "⚠️  Some alpha predictions failed. Check the logs above for details."
fi

echo "Finished at: $(date)"
exit $((EXIT_026 + EXIT_040 + EXIT_060)) 