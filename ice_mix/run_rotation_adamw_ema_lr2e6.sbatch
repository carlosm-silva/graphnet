#!/bin/bash
#SBATCH -JIceMixAugRot-AdamWEMA-2e6
#SBATCH -Agts-pli77-ideas_l40s
#SBATCH -N1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:8
#SBATCH -qinferno
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-Augmented-Rotation-AdamWEMA-2e6-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cfilho3@gatech.edu
#SBATCH --exclude=atl1-1-01-010-35-0

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
SOURCE_CKPT="ice_mix/outputs/IceMix-Augmented-Rotation_2026-05-21_11-05-56_job-8770982/checkpoints/last.ckpt"
test -f "$SOURCE_CKPT" || { echo "[FATAL] Missing source checkpoint: $SOURCE_CKPT"; exit 44; }
srun --cpu-bind=cores torchrun --nproc_per_node="${NPROC}" ice_mix/train.py data=augmented_rotation \
  project_name=IceMix-Augmented-Rotation-AdamWEMA-LR2e-6 wandb_project=IceMix-Augmented-Rotation-AdamW-EMA-Pilot wandb_group=rotation-adamw-ema-lr-sweep \
  fine_tune_from_ckpt="$SOURCE_CKPT" fine_tune.train_last_n_blocks=1 optimizer.name=adamw optimizer.adamw.eps=1e-5 optimizer.adamw.weight_decay=0.01 \
  ema.enabled=true ema.decay=0.999 lr=2e-6 use_scheduler=true scheduler.name=cosine scheduler.eta_min_factor=0.1 \
  attention.dropout=0.0 attention.attn_drop=0.0 attention.proj_drop=0.0 attention.drop_path_rate=0.0 \
  +data.rotation_seed=42 data.batch_size=32 precision=16-mixed max_epochs=8 early_stopping_patience=3 fail_on_non_finite=true
