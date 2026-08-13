# Tutorial 6 — Model changes and experimental fine-tuning

**Goal:** make one scientifically motivated model/task change safely, or reproduce the recent fine-tuning experiments without mistaking them for the default workflow.

**Prerequisites:** successful base smoke and production runs, understanding of tensor layouts, and a clean feature branch.

**Expected duration:** hours for code/test work plus **UNVERIFIED-CLUSTER** queue/training time.

**Run on:** edit/test small pieces locally or on a suitable development allocation; train on Phoenix.

**Verification:** extension points and fine-tune mechanics are **VERIFIED-STATIC**. Scientific benefit and full DDP reliability are **UNVERIFIED-CLUSTER**.

## 1. Choose one change boundary

- Pulse representation or pairwise bias: `src/models/transformer.py` and GraphNeT encoders.
- Attention/MLP behavior: `src/models/layers.py`.
- Target/output/loss: `train.py`, GraphNeT label/task classes, and `src/metrics_logging.py` together.
- Augmentation/splitting: `src/utils.py` plus data configs.
- Architecture size/regularization only: Hydra overrides, with no source edit.

Write the expected input/output shapes and physical invariance before coding.

## 2. Characterize current behavior

Run the curated maintained tests in the compatible environment:

```bash
PYTHONPATH=. python -m pytest -q \
    ice_mix/test_token_drop.py \
    ice_mix/test_ema_model.py \
    ice_mix/test_lbfgs_model.py
python -m compileall -q ice_mix
```

Add a focused test for the new invariant. Do not reorganize the package while changing scientific behavior.

`test_fast_splits.py`, `test_labels_layout.py`, and `test_splits.py` are
author-confirmed abandoned PACE probes. They now keep their hard-coded database
access behind `if __name__ == "__main__"`, so pytest collection cannot query
another user's storage, but they are not part of the maintained test claim.

Expected result for the pinned environment is 21 passing maintained tests.
This exact command passed in the disposable local environment on 2026-08-13
(**VERIFIED-LOCAL**); it does not exercise GraphNeT data loading, CUDA, or DDP.

## 3. Resolve config and smoke test

Use a Hydra override rather than editing source for run-varying choices:

```bash
python ice_mix/verify_config.py --cfg job --resolve \
    attention.hidden_dim=256 attention.depth=8
```

Then adapt the [docs-only smoke command](../examples/README.md). A two-batch smoke validates wiring only.

## 4. Understand LBFGS fine-tuning

The LBFGS launchers:

- load weights from the latest matching base run;
- freeze the backbone except its final transformer block and keep task heads trainable;
- disable stochastic dropout/drop-path so repeated closures see the same function;
- force FP32 and disable the scheduler;
- use deterministic per-batch token-drop seeds;
- synchronize scalar loss values across ranks.

The fine-tune launchers remain experimental and use automatic source-run
selection. Inspect and pin their source checkpoint before relying on them.
`run_fine_tune_smoke.sbatch` requests two L40S GPUs; the 5,000-step drop
“smoke” is a production-like stress pilot, not a quick laptop test.

## 5. Understand AdamW+EMA fine-tuning

The recent rotation pilot launchers compare learning rates while:

- loading a base rotation checkpoint into a new model;
- training the last transformer block and task head with AdamW;
- maintaining an FP32 EMA with decay 0.999;
- validating and selecting checkpoints using EMA weights;
- using cosine scheduling and mixed precision.

`predict.py` detects EMA state and uses the averaged weights. Resume must restore both Lightning state and W&B identity.

## 6. Treat fine-tuning as experimental

The original author reports some success but does not consider either fine-tuning family the go-to path. Record the source checkpoint, exact overrides, trainable parameter count, optimizer state behavior, and whether metrics use online or EMA weights. Compare against an untouched base checkpoint on the same validation events.

## Common failures

- **Changed output dimension:** update task, loss-shape validation, prediction columns, plots, and checkpoint compatibility together.
- **LBFGS nondeterminism:** remove dropout/augmentation randomness or ensure it is fixed across every closure.
- **Frozen wrong parameters:** inspect logged trainable/total count and enumerate names before a long job.
- **EMA mismatch:** distinguish plain, full EMA, and extracted inference state dictionaries.
- **Fine-tune overfitting:** compare full per-event metrics and held-out performance, not only training loss.

## What to try next

Document the experiment and its result in the group record, send the author a Slack message if historical intent is unclear, and promote the workflow to “recommended” only after a reproducible Phoenix run and review.
