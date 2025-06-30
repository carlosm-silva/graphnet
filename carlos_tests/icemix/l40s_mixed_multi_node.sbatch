#!/bin/bash
#SBATCH -JDeepIceL40sMultiNode          # Job name
#SBATCH -Agts-itaboada3                 # Charge account
#SBATCH -N3                             # Number of nodes
#SBATCH --ntasks-per-node=1             # One task (torchrun) per node
#SBATCH --cpus-per-task=32              # CPUs per task (may need adjustment, but keep for now as it's from original)
#SBATCH --gres=gpu:l40s:8               # Number of GPUs per node
#SBATCH -qembers                        # QOS Name
#SBATCH --mem=0                         # Memory per gpu (0 means all available)
#SBATCH --time=8:00:00                  # Duration of the job (Ex: 120 mins)
#SBATCH -oReport-multi-node-%j.out      # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL      # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu  # e-mail address for notifications
#SBATCH --exclude=atl1-1-03-004-31-0    # Exclude faulty node

# --- Set working directory ---------------------------------------------------
cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix    # Change to your working directory

# --- Load modules and activate conda environment -----------------------------
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1                                               # Load module dependencies
conda activate graphnet

# --- Set environment variables for multi-node training -----------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set network interface for PyTorch distributed communication
# Try different interfaces if eth0 doesn't work
export GLOO_SOCKET_IFNAME=ens10f0
export NCCL_SOCKET_IFNAME=ens10f0

# Increase PyTorch distributed timeouts
export TORCH_DISTRIBUTED_TIMEOUT=3600  # 1 hour
export NCCL_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1

# NCCL configuration for better stability
export NCCL_DEBUG=WARN  # Reduced from INFO to WARN for better performance
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10

# PyTorch distributed backend settings
export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_NCCL_TIMEOUT_MS=600000  # 10 minutes

# Additional stability settings
export NCCL_ASYNC_ERROR_HANDLING=1
# Debug variables disabled for performance
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  
# export TORCH_CPP_LOG_LEVEL=INFO

# Disable problematic features if needed
# export NCCL_P2P_DISABLE=1 # Uncomment if P2P causes issues
# export NCCL_IB_DISABLE=1  # Uncomment if InfiniBand causes issues

# Get the first node's hostname for MASTER_ADDR
# This assumes that the first node in SLURM_NODELIST is the master
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR

# Set a common port for communication
export MASTER_PORT=29501

# --- Create output directory ------------------------------------------------
export OUTPUT_DIR_BASE="/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix/telemetry_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR_BASE"

# --- Print job information ---------------------------------------------------
echo "Logs → $OUTPUT_DIR_BASE"
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Job queue: $SLURM_JOB_QUEUE"
echo "Job partition: $SLURM_JOB_PARTITION"
echo "Job nodes: $SLURM_JOB_NODELIST"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"

# --- Network diagnostics ------------------------------------------------------
echo "=== Network Diagnostics ==="
echo "Available network interfaces:"
ip addr show | grep -E "^[0-9]+:|inet "
echo "Testing connectivity to master node..."
if ping -c 3 $MASTER_ADDR; then
    echo "✓ Can ping master node"
else
    echo "✗ Cannot ping master node"
fi

# Test if master port is accessible
echo "Testing port connectivity..."
timeout 10 bash -c "</dev/tcp/$MASTER_ADDR/$MASTER_PORT" && echo "✓ Port $MASTER_PORT accessible" || echo "✗ Port $MASTER_PORT not accessible"

# --- Start GPU and system monitors on all nodes ------------------------------
echo "Starting GPU and system monitors on all nodes..."
# Monitoring now handled inside run_training.sh to avoid resource contention
echo "Monitoring will be started from within training nodes"

# No wait needed since no monitoring to initialize
echo "Proceeding directly to training..."

# --- Run the job -------------------------------------------------------------
echo "Training started..."
echo "=== Launching distributed training ==="
echo "Command: srun --nodes=$SLURM_NNODES --ntasks-per-node=1 --gres=gpu:l40s:8 bash run_training.sh"

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 --gres=gpu:l40s:8 \
     --output="${OUTPUT_DIR_BASE}/training_node%n.out" \
     --error="${OUTPUT_DIR_BASE}/training_node%n.err" \
     bash run_training.sh
EXIT_CODE=$?

# ───────────────────────── Cleanup ───────────────────────────────────────────
echo "Training finished with code $EXIT_CODE"
# No monitoring jobs to stop since they run within training nodes
echo "Monitoring stopped automatically with training"

echo "All done at $(date)"
exit $EXIT_CODE 