#!/bin/bash
#SBATCH -JIceMixAugRot-AdamWEMA-2e5-Resume
#SBATCH -Agts-pli77-ideas_l40s
#SBATCH -N1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:8
#SBATCH -qinferno
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/logs/sbatch_reports/Report-Augmented-Rotation-AdamWEMA-2e5-Resume-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cfilho3@gatech.edu
#SBATCH --exclude=atl1-1-01-010-35-0

bash /storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/run_rotation_adamw_ema_resume_common.sh IceMix-Augmented-Rotation-AdamWEMA-LR2e-5 2e-5
