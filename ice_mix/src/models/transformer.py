"""Implementation of IceMix architecture used in.

                    IceCube - Neutrinos in Deep Ice
Reconstruct the direction of neutrinos from the Universe to the South Pole

Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Set

from .layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoder,
    MahalanobisEncoder,
)
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class IceMix(GNN):
    """Encode a variable-length IceCube pulse set as one event embedding.

    IceMix is a transformer despite inheriting GraphNeT's historical ``GNN``
    interface. It Fourier-encodes normalized pulse features, applies pairwise
    spacetime-relative blocks, prepends a learned class token, and applies
    ordinary transformer blocks. The baseline does not consume graph edges.

    Input pulse order, inherited normalization, numeric encoder equations, and
    task-output semantics are defined in ``docs/numeric-contract.md``.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Dict[str, Any] = None,
        n_features: int = 6,
        maha_encoder: bool = False,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        token_drop: float = 0.0,
        drop_chance: float = 1.0,
        pos_time_multiplier: float = 4096.0,
        charge_rde_multiplier: float = 1024.0,
        spacetime_distance_scale: float = 18.0,
        spacetime_distance_clip_min: float = -4.0,
        spacetime_distance_clip_max: float = 4.0,
        spacetime_distance_multiplier: float = 1024.0,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        n_freq: float = 10000.0,
    ):
        """Construct `IceMix`.

        Args:
            hidden_dim: The latent feature dimension.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
            maha_encoder: Whether to use MahalanobisEncoder instead of
                SpacetimeEncoder for relative position encoding.
            dropout: Dropout probability for MLP layers.
            attn_drop: Dropout probability for attention weights.
            proj_drop: Dropout probability for attention projection layers.
            drop_path_rate: Maximum drop path rate. Will be scaled linearly
                from 0.0 to drop_path_rate across the depth of the model.
            token_drop: Dropout probability for input tokens.
            drop_chance: Probability that an event will have tokens dropped.
            pos_time_multiplier: Multiplier applied to normalized position and
                time before their sinusoidal encoding.
            charge_rde_multiplier: Multiplier applied to normalized charge and
                relative DOM efficiency before sinusoidal encoding.
            spacetime_distance_scale: Multiplier on normalized pairwise time
                differences in the relative spacetime or Mahalanobis interval.
            spacetime_distance_clip_min: Lower bound applied to the signed
                pairwise distance before sinusoidal encoding.
            spacetime_distance_clip_max: Upper bound applied to the pairwise
                distance before sinusoidal encoding.
            spacetime_distance_multiplier: Multiplier applied after clipping
                the pairwise distance and before sinusoidal encoding.
            mlp_ratio: Hidden-width expansion ratio in transformer MLPs.
            init_values: Initial learned residual-branch scale; ``None`` omits
                learned branch scaling.
            n_freq: Geometric frequency-range parameter used by every
                sinusoidal encoder.
        """
        super().__init__(seq_length, hidden_dim)
        self.token_drop = token_drop
        self.drop_chance = drop_chance
        self.token_drop_seed: Optional[int] = None
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_ext = FourierEncoder(
            seq_length,
            fourier_out_dim,
            scaled=scaled_emb,
            n_features=n_features,
            pos_time_multiplier=pos_time_multiplier,
            charge_rde_multiplier=charge_rde_multiplier,
            n_freq=n_freq,
        )
        if maha_encoder:
            self.rel_pos = MahalanobisEncoder(
                head_size,
                spacetime_distance_scale=spacetime_distance_scale,
                spacetime_distance_clip_min=spacetime_distance_clip_min,
                spacetime_distance_clip_max=spacetime_distance_clip_max,
                spacetime_distance_multiplier=spacetime_distance_multiplier,
                n_freq=n_freq,
            )
        else:
            self.rel_pos = SpacetimeEncoder(
                head_size,
                spacetime_distance_scale=spacetime_distance_scale,
                spacetime_distance_clip_min=spacetime_distance_clip_min,
                spacetime_distance_clip_max=spacetime_distance_clip_max,
                spacetime_distance_multiplier=spacetime_distance_multiplier,
                n_freq=n_freq,
            )
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    drop_path=(
                        drop_path_rate * (i / max(1, depth_rel - 1))
                        if depth_rel > 1
                        else 0.0
                    ),
                )
                for i in range(depth_rel)
            ]
        )
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path_rate * (i / max(1, depth - 1)) if depth > 1 else 0.0
                    ),
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.n_rel = n_rel

        if include_dynedge and dynedge_args is None:
            self.warning_once("Running with default DynEdge settings")
            self.dyn_edge = DynEdge(
                nb_inputs=9,
                nb_neighbours=9,
                post_processing_layer_sizes=[336, hidden_dim // 2],
                dynedge_layer_sizes=[
                    (128, 256),
                    (336, 256),
                    (336, 256),
                    (336, 256),
                ],
                global_pooling_schemes=None,
                activation_layer="gelu",
                add_norm_layer=True,
                skip_readout=True,
            )
        elif include_dynedge and not (dynedge_args is None):
            self.dyn_edge = DynEdge(**dynedge_args)

        self.include_dynedge = include_dynedge

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """Return parameter names excluded from weight decay.

        Returns
        -------
        set of str
            The learned class-token projection weight name.
        """
        return {"cls_token"}

    def set_token_drop_seed(self, seed: int) -> None:
        """Fix token-drop randomness for the current optimizer step.

        Parameters
        ----------
        seed : int
            Batch-specific seed assigned by ``TokenDropSeedCallback``.
        """
        self.token_drop_seed = seed

    def _token_drop_generator(
        self, device: torch.device
    ) -> Optional[torch.Generator]:
        """Return a device-local generator seeded for the current batch.

        ``None`` delegates randomness to PyTorch's global generator. A concrete
        generator makes repeated LBFGS closures and robustness evaluations use
        the same event and token masks.
        """
        if self.token_drop_seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(self.token_drop_seed)
        return generator

    def forward(self, data: Data) -> Tensor:
        """Encode pulse sets into one fixed-width vector per event.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Batched event graphs. ``data.x`` has shape
            ``(total_pulses, input_features)`` and ``data.batch`` maps each
            pulse to an event. Graph edges are not consumed by the baseline.

        Returns
        -------
        torch.Tensor
            Class-token embeddings with shape ``(n_events, hidden_dim)``.

        Notes
        -----
        During training, or when ``force_token_drop`` is set, pulse rows may be
        removed before padding. At least one pulse is retained per event.
        """
        x_input = data.x
        batch_input = data.batch
        dropped_data = data

        if (
            self.training or getattr(self, "force_token_drop", False)
        ) and self.token_drop > 0.0 and self.drop_chance > 0.0:
            batch_size = int(batch_input.max().item() + 1)
            generator = self._token_drop_generator(x_input.device)
            event_drop_mask = (
                torch.rand(
                    batch_size,
                    device=x_input.device,
                    generator=generator,
                )
                < self.drop_chance
            )
            token_in_dropped_event = event_drop_mask[batch_input]
            token_drop_mask = (
                torch.rand(
                    x_input.shape[0],
                    device=x_input.device,
                    generator=generator,
                )
                < self.token_drop
            )
            keep_mask = ~(token_in_dropped_event & token_drop_mask)

            kept_per_event = torch.bincount(
                batch_input[keep_mask], minlength=batch_size
            )
            for event_idx in torch.where(kept_per_event == 0)[0]:
                first_token_idx = torch.where(batch_input == event_idx)[0][0]
                keep_mask[first_token_idx] = True

            x_input = x_input[keep_mask]
            batch_input = batch_input[keep_mask]

            if self.include_dynedge:
                dropped_data = data.clone()
                dropped_data.x = x_input
                dropped_data.batch = batch_input
                if hasattr(dropped_data, "n_pulses"):
                    dropped_data.n_pulses = torch.bincount(
                        batch_input, minlength=batch_size
                    ).to(dropped_data.n_pulses.dtype)

        x0, mask, seq_length = array_to_sequence(
            x_input, batch_input, padding_value=0
        )
        x = self.fourier_ext(x0, seq_length)
        rel_pos_bias = self.rel_pos(x0)
        batch_size = mask.shape[0]
        
        if self.include_dynedge:
            graph = self.dyn_edge(dropped_data)
            graph, _ = to_dense_batch(graph, batch_input)
            x = torch.cat([x, graph], 2)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device),
                mask,
            ],
            1,
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]
