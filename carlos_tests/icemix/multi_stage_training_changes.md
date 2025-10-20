# Multi-Stage Training Implementation Summary

## Overview
This document summarizes the modifications made to `train_icemix_mixed.py` to implement multi-stage training with OneCycleLR scheduler and Stochastic Weight Averaging (SWA).

## Key Changes

### 1. OneCycleLR Scheduler Implementation
- **Replaced** `ReduceLROnPlateau` with `OneCycleLR`
- **Configuration**:
  - `anneal_strategy`: 'cos' (cosine annealing)
  - `pct_start`: 0.01 (1% of training for warmup)
  - `div_factor`: 25.0 (initial lr = max_lr/25)
  - `final_div_factor`: 25.0
- **Batch-level updates**: Scheduler is stepped after every optimizer step (respecting gradient accumulation)

### 2. Multi-Stage Training Procedure
The training is divided into 5 stages with progressively decreasing learning rates:

| Stage | Max LR    | SWA Active |
|-------|-----------|------------|
| 1     | 1e-4      | No         |
| 2     | 1e-5      | Yes        |
| 3     | 0.5e-5    | Yes        |
| 4     | 0.35e-5   | Yes        |
| 5     | 1e-6      | Yes        |

- Each stage trains for **8 epochs by default** (configurable)
- If `max_epochs` is less than `8 * num_stages`, epochs are divided equally
- After each stage, the best checkpoint is saved
- Subsequent stages load the best checkpoint from the previous stage

### 3. Stochastic Weight Averaging (SWA)
- **Activation**: Stage 2 onwards
- **Implementation**:
  - SWA model initialized at the start of Stage 2
  - Weights updated after each optimizer step
  - Batch normalization statistics updated after all training
  - Final SWA model saved as `final_swa_model.ckpt`

### 4. Custom Training Loop
- Created `custom_train_loop()` function for single-stage training
- Created `multi_stage_training()` function to orchestrate all stages
- Handles:
  - Progress tracking with tqdm
  - Validation after each epoch
  - Best checkpoint saving per stage
  - Learning rate logging

### 5. Checkpoint Management
- Stage-specific checkpoints: `stage_{N}_best-epoch={E}-val_loss={L}.ckpt`
- Final SWA checkpoint: `final_swa_model.ckpt`
- Prediction mode prioritizes SWA checkpoint if available

## Usage

### Training
```bash
# Default: 5 stages with 8 epochs each (40 total epochs)
python train_icemix_mixed.py --mode train --batch-size 16

# Custom total epochs (will use 8 epochs per stage if max_epochs >= 40)
python train_icemix_mixed.py --mode train --max-epochs 100 --batch-size 16

# Custom epochs per stage
python train_icemix_mixed.py --mode train --max-epochs 100 --stage-epochs 10 8 6 4 2
```

### Prediction
```bash
python train_icemix_mixed.py --mode predict
```
The script will automatically use the SWA model if available, otherwise the best regular checkpoint.

## Benefits
1. **Better convergence**: OneCycleLR with cosine annealing helps achieve better final performance
2. **Improved generalization**: SWA averages weights from multiple points, reducing overfitting
3. **Systematic training**: Multi-stage approach allows fine-tuning with progressively lower learning rates
4. **Automatic checkpoint management**: Best models are saved and loaded automatically between stages
