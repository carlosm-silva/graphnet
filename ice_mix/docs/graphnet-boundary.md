# GraphNeT interface boundary

IceMix is not a standalone data framework. It supplies the transformer and a few training extensions; GraphNeT owns the conversion from IceCube SQLite rows to event graphs and the outer reconstruction model.

## Objects IceMix configures

| GraphNeT interface | IceMix expectation |
|---|---|
| `IceCube86` | Normalizes or standardizes detector features consistently with GraphNeT's IceCube convention. |
| `IceMixNodes` | Receives the GraphNeT feature list, caps each event at `max_pulses=256`, uses `sensor_pos_z`, disables HLC-prioritized selection, and does not add ice-property columns. |
| `GraphDefinition` | Builds `Data` without persistent edges for training. The transformer only uses `x` and `batch`. |
| `KNNGraph` | Used by prediction/robustness for backward compatibility; its edges are unused by the transformer. |
| `SQLiteDataset` | Reads `SRTInIcePulses`, `truth`, the shared feature/truth names, labels, and selections from one or more database files. |
| `GraphNeTDataModulecustom` | Accepts one selection list per database, constructs train/validation/test loaders, and may combine loaders for multiple datasets. |
| `JointLabel` | Produces target order `[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]` from position plus zenith/azimuth. |
| `JointPositionandDirectionReco` | Maps the 384-channel event embedding to `[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, kappa]`. |
| `EuclideanDistanceLoss` | Returns vertex distance; IceMix metrics label it metres. |
| `VonMisesFisher3DLoss` | Scores predicted unit direction and concentration against the target direction. |
| `JointLoss` | Combines position and direction losses; IceMix subclasses it only to retain per-event values for logging. |

All table entries describe calls or configuration visible in the repository and are **VERIFIED-STATIC**. Staged schemas were inspected locally, but physical units come from the author-supplied sample manifest and still require authoritative provenance confirmation from Jiyuan.

IceMix is a transformer, not a graph neural network despite subclassing GraphNeT's historical `GNN` interface. Graph edges and the seventh `pmt_area` feature are retained for backward compatibility but unused by the backbone. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

## Objects IceMix subclasses or calls internally

- `IceMix(GNN)` uses the GraphNeT backbone contract: declare input/output widths and implement `forward(Data) -> Tensor`.
- `EMAStandardModel(StandardModel)` changes optimizer parameter selection, validation weights, checkpoint metadata, and source-checkpoint loading.
- `DistributedLBFGSStandardModel(StandardModel)` changes optimizer construction and Lightning training/optimizer hooks for repeated closures.
- `FourierEncoder` maps dense pulse features into the hidden width.
- `SpacetimeEncoder` or `MahalanobisEncoder` produces pairwise relative attention features.
- `array_to_sequence` changes flattened pulses plus `batch` IDs into padded event sequences and a validity mask.
- Optional `DynEdge` produces additional pulse features when `include_dynedge=true`; the baseline disables it.

The inherited detector transformations and encoder equations are part of the
IceMix boundary because changing them invalidates checkpoint meaning. See
[Numeric feature, encoder, and loss contract](numeric-contract.md) for the exact
feature order, normalization, Fourier/spacetime equations, and joint objective.

## Compatibility warning

The repository's pinned GraphNeT checkout contains `IceMixNodes`,
`GraphNeTDataModulecustom`, and extended encoder arguments. IceMix deliberately
does not declare a substitutable public `graphnet>=1.0.0` dependency because
those interfaces are checkout-specific. **VERIFIED-STATIC.**

For a new environment:

1. Rebuild the [checked-in exported environment](graphnet_env/README.md) at GraphNeT commit `4394131647b4a581e7d4923361b2814ab9e03ff5`.
2. Confirm these imports before submitting a GPU job:

   ```bash
   python -c 'from graphnet.models.graphs.nodes import IceMixNodes; from graphnet.data.datamodule import GraphNeTDataModulecustom'
   ```

3. Run `python ice_mix/verify_config.py --cfg job --resolve` with `DATA_ROOT` set.
4. Run the reduced smoke example before the production launch.

These imports, resolved configuration, strict checkpoint load, and a one-event
CPU forward were **VERIFIED-LOCAL** in the disposable documentation environment
on 2026-08-13. Recreating the exported stack and running on an L40S remain
**UNVERIFIED-CLUSTER**.

## What is deliberately not documented here

GraphNeT's installation, general dataset APIs, detector support, graph model catalog, and deployment tools belong to the GraphNeT project. Consult its version-matched documentation. This handoff documents only the assumptions IceMix makes at the boundary.
