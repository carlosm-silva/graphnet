#!/bin/bash

echo "=== Starting Multi-Node Training ==="
echo "Nodes: $SLURM_NNODES | Job ID: $SLURM_JOB_ID"

# Set environment variables
export MASTER_PORT=29501

# Get the actual master node name
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Start monitoring on this node (in background)
echo "Starting local monitoring on node $SLURM_NODEID..."

# Ensure OUTPUT_DIR_BASE is available for monitoring scripts
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-/tmp/telemetry_${SLURM_JOB_ID}}"

# Start GPU monitoring in background
if [ -f "monitor_gpu.sh" ]; then
    bash monitor_gpu.sh > "${OUTPUT_DIR_BASE}/gpu_monitor_node${SLURM_NODEID}.log" 2>&1 &
    GPU_MON_PID=$!
    echo "GPU monitoring started (PID: $GPU_MON_PID)"
else
    echo "Warning: monitor_gpu.sh not found"
fi

# Start system monitoring in background  
if [ -f "monitor_system.py" ]; then
    python monitor_system.py > "${OUTPUT_DIR_BASE}/system_monitor_node${SLURM_NODEID}.log" 2>&1 &
    SYS_MON_PID=$!
    echo "System monitoring started (PID: $SYS_MON_PID)"
else
    echo "Warning: monitor_system.py not found"
fi

# Cleanup function to stop monitoring on exit
cleanup() {
    echo "Stopping monitoring on node $SLURM_NODEID..."
    [ ! -z "$GPU_MON_PID" ] && kill $GPU_MON_PID 2>/dev/null
    [ ! -z "$SYS_MON_PID" ] && kill $SYS_MON_PID 2>/dev/null
    echo "Monitoring stopped"
}

# Set up trap to ensure cleanup on exit
trap cleanup EXIT

echo "Starting training..."

# Run torchrun
torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=8 \
         --node_rank=$SLURM_NODEID \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         train_icemix_mixed_multi_node.py --max-epochs 8 --early-stopping-patience 150 --batch-size 128 --num-workers 3 