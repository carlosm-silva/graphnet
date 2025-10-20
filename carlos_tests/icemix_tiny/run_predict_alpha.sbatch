#!/bin/bash
#SBATCH -J predict_alpha
#SBATCH -Agts-itaboada3
#SBATCH -qembers
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:3
#SBATCH --time=01:00:00
#SBATCH --output=predict_alpha_%j.out
#SBATCH --error=predict_alpha_%j.err
#SBATCH --mail-user=cfilho3@gatech.edu	# e-mail address for notifications

# Load the anaconda module
module load anaconda3/2022.05.0.1

# Activate the graphnet conda environment
conda activate graphnet

# Run the prediction script
bash /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/run_predict_alpha.sh
