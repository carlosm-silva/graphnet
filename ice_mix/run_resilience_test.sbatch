#!/bin/bash
#SBATCH -JIceMixResilience                       # Job name
#SBATCH -Agts-pli77-ideas_l40s                   # Charge account
#SBATCH -N1                                      # Number of nodes
#SBATCH --gres=gpu:l40s:1                        # GPUs required (1 is sufficient for inference usually)
#SBATCH -qinferno                                # QOS Name
#SBATCH --mem=128G                               # Memory per gpu (0 means all available)
#SBATCH --time=2:00:00                           # Duration
#SBATCH -o IceMixResilience-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --tmp=3000G                              # Temporary directory (3TB)

# Supported seeded token-removal evaluation. Submit from the checkout root.
set -u
if [ -z "${SLURM_SUBMIT_DIR:-}" ] || [ ! -f "$SLURM_SUBMIT_DIR/ice_mix/resilience_test.py" ]; then
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
TEST_FRACTION="${ICE_MIX_TEST_FRACTION:-0.01}"
SEED="${ICE_MIX_SEED:-42}"
python ice_mix/resilience_test.py \
    --base-dir "$OUTPUTS_DIR" \
    --n-gpus 1 \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED"
exit $?
