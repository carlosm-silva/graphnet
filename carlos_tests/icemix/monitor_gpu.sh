#!/bin/bash

# Get the output directory from environment variable, with fallback
OUTPUT_DIR="${OUTPUT_DIR_BASE}/node_${SLURM_NODEID}"
if [ -z "$OUTPUT_DIR_BASE" ] || [ -z "$SLURM_NODEID" ]; then
    echo "Warning: OUTPUT_DIR_BASE or SLURM_NODEID not set, using fallback"
    OUTPUT_DIR="./monitoring/node_${SLURM_NODEID:-unknown}"
fi

mkdir -p "$OUTPUT_DIR"

# Run nvidia-smi dmon in the background with error handling
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi dmon -s pucvmet -d 1 -o TD -f "${OUTPUT_DIR}/nvidia_dmon.log" &
    echo "GPU monitoring started for node $SLURM_NODEID"
else
    echo "nvidia-smi not found, GPU monitoring disabled"
fi

# Keep the script running until manually killed
wait 