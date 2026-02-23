#!/bin/bash
#SBATCH -JNodeTest                         # Job name
#SBATCH -Agts-pli77-ideas_l40s                   # Charge account
#SBATCH -N1                                      # Number of nodes
#SBATCH --gres=gpu:l40s:4                        # GPUs required
#SBATCH -qinferno                                # QOS Name
#SBATCH --mem=100                                  # Memory per gpu (0 means all available)
#SBATCH --time=0:01:00                           # Duration
#SBATCH -o /storage/project/r-itaboada3-0/cfilho3/
#SBATCH --nodelist=atl1-1-01-010-31-0

python hello.py
