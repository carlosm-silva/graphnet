#!/bin/bash
#SBATCH -JIceMixDrop-FineTune-Smoke                       # Job name
#SBATCH -Agts-pli77-ideas_l40s                             # Charge account
#SBATCH -N1                                                # Number of nodes
#SBATCH --cpus-per-task=8                                  # Number of CPUs
#SBATCH --gres=gpu:l40s:2                                  # Exercise multi-GPU LBFGS
#SBATCH -qinferno                                          # QOS Name
#SBATCH --mem=0                                            # Use available node memory
#SBATCH --time=04:00:00                                    # Cover 5,000 production-like steps
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-Drop-FineTune-Smoke-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL                         # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu                     # Notifications
#SBATCH --exclude=atl1-1-01-010-35-0

cd /storage/project/r-itaboada3-0/cfilho3/graphnet
mkdir -p ice_mix/logs/sbatch_reports
source ice_mix/resume_common.sh

load_ice_mix_env
copy_standard_data_to_local_tmp || exit $?

SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1
conda activate graphnet

configure_healthy_gpus || exit $?
configure_training_environment

SOURCE_PROJECT_NAME="IceMix-Drop"
PROJECT_NAME="IceMix-Drop-FineTune-LBFGS-Smoke"
EXTRA_ARGS=()
configure_fine_tune_from_latest_run "$SOURCE_PROJECT_NAME" "drop LBFGS smoke fine-tuning" || exit $?

echo "Smoke job started at $(date) on $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

srun --cpu-bind=cores \
     torchrun --nproc_per_node="${NPROC}" \
     ice_mix/train.py \
     data=drop \
     project_name="${PROJECT_NAME}" \
     optimizer.name=lbfgs \
     optimizer.lbfgs.max_iter=20 \
     optimizer.lbfgs.history_size=20 \
     optimizer.lbfgs.line_search_fn=strong_wolfe \
     fine_tune.train_last_n_blocks=1 \
     attention.dropout=0.0 \
     attention.attn_drop=0.0 \
     attention.proj_drop=0.0 \
     attention.drop_path_rate=0.0 \
     precision=32-true \
     data.batch_size=32 \
     use_scheduler=false \
     max_epochs=1 \
     early_stopping_patience=0 \
     limit_train_batches=5000 \
     limit_val_batches=2 \
     num_workers=1 \
     wandb=true \
     fail_on_non_finite=true \
     "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
echo "Smoke job finished with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
