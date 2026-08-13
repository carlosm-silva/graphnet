# IceMix Configuration Parameters Guide

> **Historical guide:** This file predates the current Hydra schema and contains stale keys and descriptions. Use [`docs/configuration.md`](docs/configuration.md) and the checked-in YAML/source as the current handoff reference. In particular, the current config derives device count from `CUDA_VISIBLE_DEVICES`, random/CSV splits live under `data.split`, and `train.py` does not call the legacy multistage trainer. **VERIFIED-STATIC.**

This document details the configuration parameters available in the Hydra configuration setup for the `IceMix` codebase. Configurations are split across multiple files in the `conf/` directory: the primary `config.yaml`, data configurations in `conf/data/`, and model/attention architecture configurations in `conf/attention/`.

---

## 1. Global & Training Settings (`conf/config.yaml`)

These parameters govern the overall training environment, optimization procedures, and hardware utilization.

* **`project_name` & `run_name`**: Identifiers for logging and organization (e.g., in WandB).
* **`seed`**: Random seed to ensure reproducibility.
* **`debug`**: Boolean flag to enable or disable debug mode for faster local execution or testing.
* **`precision`**: Floating-point precision for PyTorch Lightning (e.g., `"16-mixed"` for mixed precision to save memory).
* **`gpus`**: List of GPU indices to utilize during training or prediction (e.g., `[0, 1]`).
* **`num_workers`**: Number of CPU subprocesses used for data loading.
* **`max_epochs`**: Maximum number of training epochs to run.
* **`early_stopping_patience`**: Number of epochs with no improvement in validation loss before training is halted automatically.
* **`accumulate_grad_batches`**: Number of batches to accumulate gradients over before performing an optimizer step. Useful for simulating larger batch sizes.
* **`wandb`**: Boolean flag to enable or disable Weights & Biases logging.
* **`lr`**: The learning rate for the optimizer (e.g., AdamW).
* **`alpha`**: Weighting parameter for `JointLoss` (balances between position loss and direction loss).
* **`use_scheduler`**: Whether to use a learning rate scheduler (e.g., `ReduceLROnPlateau`).
* **`scheduler`**: A sub-configuration containing parameters for the learning rate scheduler (e.g., `patience`, `factor`).
* **`multistage`**: Boolean flag to determine if training should be executed in multiple sequential stages.
* **`num_stages`**: Number of stages defined if multistage training is active.
* **`stage_epochs`**: Defines precise epoch allocation if multistage configurations demand custom lengths.
* **`output_dir`, `scratch_dir`, `checkpoint_dir`, `logs_dir`**: System paths where checkpoints, logs, and outputs are saved.

---

## 2. Data Loading & Processing (`conf/data/standard.yaml`)

These configurations dictate how the raw SQLite databases are read, sampled, and converted into graph objects.

* **`path`**: A list of paths pointing to the SQLite databases containing the training/validation data.
* **`pulsemap`**: The name of the specific table or pulsemap in the database to read features from (e.g., `"SRTInIcePulses"`).
* **`target`**: The target variable the model is predicting (e.g., `"direction"`).
* **`truth_table`**: The database table containing true labels for the events.
* **`batch_size`**: The number of graphs/events processed in a single batch.
* **`pin_memory` / `persistent_workers` / `prefetch_factor`**: PyTorch DataLoader optimization flags to maximize GPU data throughput.
* **`max_pulses`**: Hard cap on the maximum number of pulses processed per event/graph. If an event has more pulses, it will be truncated. (Previously a "magic number" hardcoded to `256`).
* **`train_selections` & `val_selections`**: Lists identifying which event ID selections should be dynamically loaded to build the exact training and validation sets.
* **`train_val_split`**: Ratio to split datasets if explicit selections are not provided (e.g., `[0.2, 0.8]`).

---

## 3. Model Architecture & Encoders (`conf/attention/baseline.yaml`)

These parameters define the internal structure, dimensions, and mathematical constants driving the inner workings of the `IceMix` Transformer and its relative-position encoders.

### 3.1 Structural Transformer Parameters

* **`model_name`**: String identifier specifying the overarching model class/architecture (e.g., `"IceMix"`).
* **`hidden_dim`**: The latent channel dimension inside the Transformer blocks.
* **`seq_length`**: Dimensionality of the base positional embeddings.
* **`depth`**: The total number of standard Transformer blocks to layer.
* **`head_size`**: The dimension size for individual attention heads.
* **`n_rel`**: The number of "relative" transformer layers to use (layers that apply pseudo-Lorentz / Mahalanobis geometry bias matrices over attention).
* **`scaled_emb`**: Boolean denoting whether the initial token embeddings are scaled before positional encodings are added.
* **`include_dynedge`**: Boolean to determine if graph-based convolution (`DynEdge`) should be pre-pended, augmenting the node features before transformer processing.
* **`n_features`**: The number of input features (x, y, z, time, charge, etc.) per node pulse.
* **`maha_encoder`**: If `True`, the model uses relative Mahalanobis distances (positive spatial-temporal metric). If `False`, uses `SpacetimeEncoder` (pseudo-Lorentz distance metric).

### 3.2 Regularization & Dropouts

* **`dropout`**: Linear dropout probability for MLP layers.
* **`attn_drop`**: Dropout probability for the multi-head attention weight matrices.
* **`proj_drop`**: Dropout probability for attention projection layers.
* **`drop_path_rate`**: Maximum Stochastic Depth drop rate, scaling linearly across transformer depth.
* **`token_drop`**: Probability of completely dropping an input pulse/token before it enters the transformer (useful for resilience/robustness testing).

### 3.3 Internal Mathematical Scalars ("Magic Numbers")

Historically hardcoded, these scalars manipulate the raw physics of the input measurements to better align across the functional frequencies of sinusoidal encoders or neural network expansions:

* **Positional & Feature Scales**:
  * **`pos_time_multiplier`**: Scales absolute X, Y, Z spatial coordinates and Time coordinates before they are passed into the sinusoidal generator (e.g., `4096.0`).
  * **`charge_rde_multiplier`**: Scales secondary features like Charge and Relative DOM Efficiency (RDE) before sinusoidal encoding (e.g., `1024.0`).
* **Spacetime / Geometry Metrics**:
  * **`spacetime_distance_scale`**: Multiplies differences of GraphNeT-normalized time inside the dimensionless signed spacetime interval. The historical `18.0 = 3e4 / 500 * 3e-1` is algebraically consistent with `0.3 m/ns`, but its tuning provenance is unknown and it must not be described as a measured propagation speed in deep ice. See `docs/numeric-contract.md`.
  * **`spacetime_distance_clip_min` / `max`**: Hard-caps the mathematically derived metric separation between any two pulses to prevent extreme outlier bounds (e.g., `-4.0` / `4.0`).
  * **`spacetime_distance_multiplier`**: Final scaling multiplier on the calculated pairwise distance matrix before sinusoidal projection (e.g., `1024.0`).
* **Network Size Adjustments**:
  * **`mlp_ratio`**: Defines the hidden-layer expansion size inside the Transformer's Multi-Layer Perceptrons. An `mlp_ratio` of `4.0` expands a `hidden_dim` of 384 to 1536 before bottling it back down.
  * **`init_values`**: Starting scaler value for residual connection paths in relative blocks (e.g., `1.0`).
  * **`n_freq`**: Base frequency wave limit (traditionally `10000.0` originating from the *Attention Is All You Need* paper) used to calculate sinusoidal cycle periods.
