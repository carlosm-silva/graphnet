# Model implementation

- `transformer.py` defines the `IceMix` GraphNeT backbone.
- `layers.py` defines relative-attention and ordinary transformer blocks plus older graph-convolution experiments.
- `ema_model.py` adds an FP32 exponential moving average around GraphNeT `StandardModel`.
- `lbfgs_model.py` adds DDP-safe LBFGS closures and last-block fine-tuning.
- `__init__.py` re-exports EMA helpers.

The stable dependency direction is layers → backbone → GraphNeT task/model wrapper → `train.py`. The EMA and LBFGS wrappers are mutually exclusive training modes selected in `train.py`. **VERIFIED-STATIC.**
