# Attention configurations

`baseline.yaml` defines the IceMix backbone: Fourier embedding widths, transformer depth, head size, relative-position encoder, dropout, stochastic depth, and numeric scaling constants.

The checked-in baseline is production-scale. A consumer GPU should reduce `hidden_dim`, `depth`, `n_rel`, `data.batch_size`, and `data.max_pulses`; see [`docs/examples`](../../docs/examples/README.md). **VERIFIED-STATIC.**
