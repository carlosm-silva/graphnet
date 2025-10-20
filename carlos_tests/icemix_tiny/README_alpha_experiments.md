# Alpha Experiments for IceMix Training

This directory contains scripts to train models with different alpha values using the pre-trained model as initialization.

## Overview

The experiments train models with different alpha values (0.026, 0.040, 0.060) for the joint loss function, using the pre-trained model with alpha=0 as initialization. Each experiment runs in parallel with separate checkpoint and log directories.

## Files

### Training Scripts
- `train_icemix_mixed_tiny_alpha.py` - Modified training script that supports different alpha values
- `l40s_mixed_alpha_0_026.sbatch` - SLURM script for alpha=0.026
- `l40s_mixed_alpha_0_040.sbatch` - SLURM script for alpha=0.040  
- `l40s_mixed_alpha_0_060.sbatch` - SLURM script for alpha=0.060

### Submission Scripts
- `submit_alpha_experiments.sh` - Script to submit all three experiments in parallel
- `run_predict_alpha.sh` - Script to run predictions for all alpha values

## Directory Structure

After running the experiments, the following directory structure will be created:

```
carlos_tests/icemix_tiny/
├── checkpoints/
│   ├── alpha_0_026/          # Checkpoints for alpha=0.026
│   ├── alpha_0_040/          # Checkpoints for alpha=0.040
│   └── alpha_0_060/          # Checkpoints for alpha=0.060
├── logs/
│   ├── alpha_0_026/          # Training logs for alpha=0.026
│   ├── alpha_0_040/          # Training logs for alpha=0.040
│   └── alpha_0_060/          # Training logs for alpha=0.060
└── results/
    └── [database_name]/
        ├── dynedgeTITO_direction_alpha_0_026_example/
        ├── dynedgeTITO_direction_alpha_0_040_example/
        └── dynedgeTITO_direction_alpha_0_060_example/
```

## Usage

### 1. Submit Training Jobs

To submit all three experiments in parallel:

```bash
cd carlos_tests/icemix_tiny/
./submit_alpha_experiments.sh
```

This will submit three SLURM jobs:
- Alpha 0.026: `DeepIceL40sAlpha026TinyMinkDrop`
- Alpha 0.040: `DeepIceL40sAlpha040TinyMinkDrop`  
- Alpha 0.060: `DeepIceL40sAlpha060TinyMinkDrop`

### 2. Monitor Jobs

Check job status:
```bash
squeue -u $USER
```

### 3. Run Predictions

After training is complete, run predictions for all alpha values:

```bash
./run_predict_alpha.sh
```

## Key Features

### Pre-trained Model Initialization
- All experiments use the pre-trained model from `checkpoints/last.ckpt` as initialization
- The loss function parameters are excluded from the initialization since they depend on alpha
- Only the backbone network weights are transferred

### Separate Directories
- Each alpha value has its own checkpoint directory: `checkpoints/alpha_X_XXX/`
- Each alpha value has its own log directory: `logs/alpha_X_XXX/`
- Results are saved in separate subdirectories under `results/`

### Parallel Training
- All three experiments can run simultaneously
- Each uses 8 L40S GPUs
- Independent checkpointing and logging

## Alpha Values

The experiments use the following alpha values:
- **0.026**: Lower weight for direction loss
- **0.040**: Medium weight for direction loss  
- **0.060**: Higher weight for direction loss

## Monitoring

### Training Progress
- Check individual log files in `logs/alpha_X_XXX/`
- Monitor GPU usage with `nvidia-smi`
- Check SLURM job status with `squeue`

### Results
- Training logs: `logs/alpha_X_XXX/training_logs/`
- Checkpoints: `checkpoints/alpha_X_XXX/`
- Predictions: `results/[database_name]/dynedgeTITO_direction_alpha_X_XXX_example/`

## Troubleshooting

### Job Submission Issues
- Check SLURM queue status: `squeue -q embers`
- Verify GPU availability: `sinfo -p embers`

### Training Issues
- Check log files in the telemetry directories
- Monitor GPU memory usage
- Verify data paths and permissions

### Prediction Issues
- Ensure training completed successfully
- Check checkpoint file existence
- Verify model loading in prediction mode 