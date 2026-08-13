#!/bin/bash
#SBATCH -JIceMix-Rotation-Average
#SBATCH -Agts-pli77-ideas_l40s
#SBATCH -N1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH -qinferno
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --tmp=3000G
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-Rotation-Average-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cfilho3@gatech.edu

cd /storage/project/r-itaboada3-0/cfilho3/graphnet || exit 1
mkdir -p ice_mix/logs/sbatch_reports
source ice_mix/resume_common.sh

load_ice_mix_env
copy_standard_data_to_local_tmp || exit $?

module load anaconda3/2022.05.0.1
conda activate graphnet

configure_healthy_gpus || exit $?
configure_training_environment

CHECKPOINT_DIR="ice_mix/outputs/IceMix-Augmented-Rotation-FineTune-LBFGS_2026-07-17_09-03-11_job-11230412/checkpoints"
CHECKPOINT_A="${CHECKPOINT_DIR}/best-epoch=02-val_loss=1.2792.ckpt"
CHECKPOINT_B="${CHECKPOINT_DIR}/best-epoch=06-val_loss=1.2714.ckpt"
OUTPUT_DIR="ice_mix/outputs/IceMix-Rotation-Checkpoint-Averaging_job-${SLURM_JOB_ID}"

echo "Job started at $(date) on ${SLURM_JOB_NODELIST}"
echo "Interpolating from ${CHECKPOINT_A} to ${CHECKPOINT_B}"
echo "Results will be written to ${OUTPUT_DIR}/results.csv"

srun --cpu-bind=cores \
    python ice_mix/evaluate_rotation_checkpoint_averaging.py \
    --checkpoint-a "${CHECKPOINT_A}" \
    --checkpoint-b "${CHECKPOINT_B}" \
    --lambdas 0.0 0.25 0.5 0.75 1.0 \
    --batch-size 128 \
    --num-workers 3 \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Job finished with code ${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
