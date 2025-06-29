#!/bin/bash
#SBATCH -JDeepIceRTX                                  # Job name 
#SBATCH -Agts-itaboada3                      # Charge account
#SBATCH -N1 
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:rtx_6000:4                       # Number of nodes and GPUs required
#SBATCH -qembers                                   # QOS Name
#SBATCH --mem-per-gpu=85G                           # Memory per gpu
#SBATCH --time=8:00:00                              # Duration of the job (Ex: 120 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=cfilho3@gatech.edu              # e-mail address for notifications


cd /storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix      # Change to your working directory 
SLURM_MPI_TYPE=pmix_v4
module load anaconda3/2022.05.0.1                    # Load module dependencies
conda activate graphnet 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- monitor GPU every second ----------------------------------------------
nvidia-smi dmon -s pucvmet -d 1 -o TD -f "nvidia_dmon.log" &
GPU_MON_PID=$!

export CUDA_LAUNCH_BLOCKING=0 # This is the default, set to 1 for debugging

export OMP_NUM_THREADS=6                                         #Bind CPU to GPUs
export MKL_NUM_THREADS=6                                         # Suggested by GPT, shouldn't really matter
srun --cpu-bind=cores\
     torchrun --nproc_per_node=4 train_icemix.py --max-epochs 80 --early-stopping-patience 150 --batch-size 64 --num-workers 12  # Run test example

kill $GPU_MON_PID 2>/dev/null