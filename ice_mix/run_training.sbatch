#!/bin/bash
#SBATCH -JIceMixTraining           	    # Job name 
#SBATCH -Agts-pli77-ideas_l40s			# Charge account
#SBATCH -N1 					        # Number of nodes
#SBATCH --cpus-per-task=32              # Number of CPUs per task (4 per GPU × 8 GPUs = 32 total)
#SBATCH --gres=gpu:l40s:8			# Number of nodes and GPUs required
#SBATCH -qinferno					    # QOS Name
#SBATCH --mem=0					        # Memory per gpu (0 means all available)
#SBATCH --time=72:00:00				    # Duration of the job (8 hours for full training)
#SBATCH -o IceMixTraining-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL	    # Mail preferences
#SBATCH --exclude=atl1-1-01-010-35-0



# This is the supported base launcher for all four production data variants.
# Submit it from the GraphNeT checkout root. Optional submit-time variables:
#   ICE_MIX_DATA_CONFIG=standard|augmented_rotation|drop|drop_aug_rot
#   ICE_MIX_PROJECT_NAME=<matching durable experiment label>
#   ICE_MIX_CKPT_PATH=<explicit last.ckpt to resume>
#   ICE_MIX_WANDB_RUN_ID=<explicit existing W&B ID, required with resume>

set -u

# --- Set working directory ---------------------------------------------------
if [ -z "${SLURM_SUBMIT_DIR:-}" ] || [ ! -f "$SLURM_SUBMIT_DIR/ice_mix/train.py" ]; then
    echo "[FATAL] Submit from the GraphNeT checkout root containing ice_mix/train.py."
    exit 40
fi
cd "$SLURM_SUBMIT_DIR" || exit 40

# --- Create output directory for logs if it doesn't exist --------------------
mkdir -p ice_mix/logs/sbatch_reports
source ice_mix/resume_common.sh

# --- Load .env (must precede the NVMe copy so DATA_ROOT is in scope) --------
# `set -a; source` honours quoted values, needed for paths containing spaces.
if [ -f ice_mix/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source <(grep -v '^#' ice_mix/.env)
    set +a
fi

# --- Copy all three databases to local NVMe ---------------------------------
copy_standard_data_to_local_tmp || exit $?

# --- Load modules and activate conda environment -----------------------------
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1
conda activate graphnet

python -m pip check || exit 41
python -c 'from graphnet.data.datamodule import GraphNeTDataModulecustom; from graphnet.models.graphs.nodes import IceMixNodes' || exit 41
configure_healthy_gpus || exit $?
configure_training_environment || exit $?

DATA_CONFIG="${ICE_MIX_DATA_CONFIG:-standard}"
PROJECT_NAME="${ICE_MIX_PROJECT_NAME:-IceMix}"
EXTRA_ARGS=()
case "$DATA_CONFIG" in
    standard|augmented_rotation|drop|drop_aug_rot) ;;
    *) echo "[FATAL] Unsupported ICE_MIX_DATA_CONFIG: $DATA_CONFIG"; exit 51 ;;
esac

if [ -n "${ICE_MIX_CKPT_PATH:-}" ]; then
    if [ ! -r "$ICE_MIX_CKPT_PATH" ] || [ ! -s "$ICE_MIX_CKPT_PATH" ]; then
        echo "[FATAL] Resume checkpoint is missing, unreadable, or empty: $ICE_MIX_CKPT_PATH"
        exit 53
    fi
    if [ -z "${ICE_MIX_WANDB_RUN_ID:-}" ]; then
        echo "[FATAL] Set ICE_MIX_WANDB_RUN_ID for an explicit resume."
        exit 54
    fi
    RUN_DIR=$(dirname "$(dirname "$ICE_MIX_CKPT_PATH")")
    RUN_NAME=$(basename "$RUN_DIR")
    if [ ! -f "$RUN_DIR/.hydra/config.yaml" ]; then
        echo "[FATAL] Resume run lacks .hydra/config.yaml: $RUN_DIR"
        exit 55
    fi
    export WANDB_RUN_ID="$ICE_MIX_WANDB_RUN_ID"
    export WANDB_RESUME="must"
    EXTRA_ARGS+=("ckpt_path=$ICE_MIX_CKPT_PATH" "run_name=$RUN_NAME")
    echo "Explicit resume: $ICE_MIX_CKPT_PATH (run $RUN_NAME, W&B $WANDB_RUN_ID)"
fi

echo "Preflighting Hydra configuration: data=$DATA_CONFIG project_name=$PROJECT_NAME"
python ice_mix/verify_config.py --cfg job --resolve \
    data="$DATA_CONFIG" project_name="$PROJECT_NAME" >/dev/null || exit 52

# --- Run the job -------------------------------------------------------------
echo "Job started at $(date) on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

# Run with python directly. Hydra will handle loading config.yaml
srun --cpu-bind=cores \
     torchrun --nproc_per_node=${NPROC} \
     ice_mix/train.py \
     data="${DATA_CONFIG}" \
     project_name="${PROJECT_NAME}" \
     "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
echo "Job finished with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
