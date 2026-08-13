#!/bin/bash
#SBATCH -JIceMixPredict                          # Job name
#SBATCH -Agts-pli77-ideas_l40s                   # Charge account
#SBATCH -N1                                      # Number of nodes
#SBATCH --gres=gpu:l40s:1                        # GPUs required (1 is sufficient for inference usually)
#SBATCH -qinferno                                # QOS Name
#SBATCH --mem=128G                               # Memory per gpu (0 means all available)
#SBATCH --time=14:00:00                           # Duration
#SBATCH -o IceMixPredict-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --tmp=3000G                              # Temporary directory (3TB)

# Supported prediction launcher. Submit from the GraphNeT checkout root.
# Optional variables: ICE_MIX_OUTPUTS_DIR, ICE_MIX_FORCE=1,
# ICE_MIX_USE_TEST_SPLIT=1, and ICE_MIX_WANDB_ROOT.
set -u

if [ -z "${SLURM_SUBMIT_DIR:-}" ] || [ ! -f "$SLURM_SUBMIT_DIR/ice_mix/predict.py" ]; then
    echo "[FATAL] Submit from the GraphNeT checkout root."
    exit 40
fi
cd "$SLURM_SUBMIT_DIR" || exit 40
source ice_mix/resume_common.sh
load_ice_mix_env
copy_standard_data_to_local_tmp || exit $?

module load anaconda3/2022.05.0.1
conda activate graphnet
python -m pip check || exit 41
python -c 'from graphnet.data.datamodule import GraphNeTDataModulecustom; from graphnet.models.graphs.nodes import IceMixNodes' || exit 41
configure_healthy_gpus || exit $?
configure_training_environment || exit $?

OUTPUTS_DIR="${ICE_MIX_OUTPUTS_DIR:-$PWD/ice_mix/outputs}"
if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "[FATAL] Prediction output tree does not exist: $OUTPUTS_DIR"
    exit 56
fi

PREDICT_ARGS=(--base-dir "$OUTPUTS_DIR" --n-gpus 1)
if [ "${ICE_MIX_FORCE:-0}" = "1" ]; then PREDICT_ARGS+=(--force); fi
if [ "${ICE_MIX_USE_TEST_SPLIT:-0}" = "1" ]; then PREDICT_ARGS+=(--use-test-split); fi

echo "Prediction job started at $(date) on $SLURM_JOB_NODELIST"
python ice_mix/predict.py "${PREDICT_ARGS[@]}"
EXIT_CODE=$?
echo "Prediction job finished with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
