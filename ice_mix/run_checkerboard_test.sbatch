#!/bin/bash
#SBATCH -JIceMixCheckerboard                     # Job name
#SBATCH -Agts-pli77-ideas_l40s                   # Charge account
#SBATCH -N1                                      # Number of nodes
#SBATCH --gres=gpu:l40s:1                        # GPUs required (1 is sufficient for inference usually)
#SBATCH -qinferno                                # QOS Name
#SBATCH --mem=128G                               # Memory per gpu (0 means all available)
#SBATCH --time=12:00:00                          # Duration
#SBATCH -o IceMixComplementaryHalf-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --tmp=3000G                              # Temporary directory (3TB)

# Supported historical complementary-half evaluation (not spatial). Set
# ICE_MIX_RUN_DIR to pin an exact run instead of selecting by project/loss.
set -u
if [ -z "${SLURM_SUBMIT_DIR:-}" ] || [ ! -f "$SLURM_SUBMIT_DIR/ice_mix/checkerboard_test.py" ]; then
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
configure_healthy_gpus || exit $?
configure_training_environment || exit $?

OUTPUTS_DIR="${ICE_MIX_OUTPUTS_DIR:-$PWD/ice_mix/outputs}"
MODEL_CONFIG="${ICE_MIX_MODEL_CONFIG:-IceMix}"
TEST_FRACTION="${ICE_MIX_TEST_FRACTION:-1.0}"
SEED="${ICE_MIX_SEED:-42}"
CHECKERBOARD_ARGS=(
    --model-config "$MODEL_CONFIG"
    --n-gpus 1
    --base-dir "$OUTPUTS_DIR"
    --test-fraction "$TEST_FRACTION"
    --seed "$SEED"
)
if [ -n "${ICE_MIX_RUN_DIR:-}" ]; then
    CHECKERBOARD_ARGS+=(--run-dir "$ICE_MIX_RUN_DIR")
fi
python ice_mix/checkerboard_test.py \
    "${CHECKERBOARD_ARGS[@]}"
exit $?
