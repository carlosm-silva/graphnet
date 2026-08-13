#!/bin/bash
# Shared body for the three AdamW+EMA pilot resume jobs.
PROJECT_NAME="$1"
LEARNING_RATE="$2"

cd /storage/project/r-itaboada3-0/cfilho3/graphnet || exit $?
mkdir -p ice_mix/logs/sbatch_reports
source ice_mix/resume_common.sh
load_ice_mix_env
copy_standard_data_to_local_tmp || exit $?
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1
conda activate graphnet
configure_healthy_gpus || exit $?
configure_training_environment

EXTRA_ARGS=()
configure_fine_tune_resume_from_latest_run "$PROJECT_NAME" "rotation AdamW+EMA pilot" || exit $?
srun --cpu-bind=cores torchrun --nproc_per_node="${NPROC}" ice_mix/train.py data=augmented_rotation \
  project_name="$PROJECT_NAME" wandb_project=IceMix-Augmented-Rotation-AdamW-EMA-Pilot wandb_group=rotation-adamw-ema-lr-sweep \
  fine_tune.train_last_n_blocks=1 optimizer.name=adamw optimizer.adamw.eps=1e-5 optimizer.adamw.weight_decay=0.01 \
  ema.enabled=true ema.decay=0.999 lr="$LEARNING_RATE" use_scheduler=true scheduler.name=cosine scheduler.eta_min_factor=0.1 \
  attention.dropout=0.0 attention.attn_drop=0.0 attention.proj_drop=0.0 attention.drop_path_rate=0.0 \
  +data.rotation_seed=42 data.batch_size=32 precision=16-mixed max_epochs=8 early_stopping_patience=3 fail_on_non_finite=true \
  "${EXTRA_ARGS[@]}"
