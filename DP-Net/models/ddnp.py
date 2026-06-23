"""Data-driven neural path of DP-Net."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import (
    CrossElectrodeSpatialFilter,
    GatedEmbedding,
    MultiScaleTemporalModule,
    PositionalEncoding,
    TemporalFeatureHead,
    TemporalTransformerBlock,
)


class DataDrivenNeuralPath(nn.Module):
    """DDNP: gated spatial encoding, temporal attention, and multi-scale modeling."""

    def __init__(
        self,
        num_electrodes: int,
        time_points: int,
        gated_embedding_dim: int = 32,
        spatial_dim: int = 64,
        temporal_kernel_sizes: tuple[int, ...] = (31, 15, 7, 1),
        attention_heads: int = 8,
        transformer_feedforward_dim: int = 256,
        transformer_dropout: float = 0.0,
        temporal_dropout: float = 0.0,
        temporal_head_hidden_dim: int = 16,
        temporal_head_output_dim: int = 4,
        use_mstm: bool = True,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_electrodes = num_electrodes
        self.time_points = time_points
        self.use_mstm = use_mstm
        self.use_self_attention = use_self_attention

        self.gated_embedding = GatedEmbedding(num_electrodes, gated_embedding_dim)
        self.spatial_filter = CrossElectrodeSpatialFilter(
            in_channels=gated_embedding_dim,
            out_channels=spatial_dim,
            num_electrodes=num_electrodes,
        )
        self.positional_encoding = PositionalEncoding(spatial_dim, time_points)
        self.temporal_transformer = TemporalTransformerBlock(
            embedding_dim=spatial_dim,
            num_heads=attention_heads,
            feedforward_dim=transformer_feedforward_dim,
            dropout=transformer_dropout,
        )
        self.multi_scale_temporal = MultiScaleTemporalModule(
            in_channels=num_electrodes,
            out_channels=spatial_dim,
            kernel_sizes=temporal_kernel_sizes,
            dropout=temporal_dropout,
        )
        self.temporal_head = TemporalFeatureHead(
            input_dim=spatial_dim,
            hidden_dim=temporal_head_hidden_dim,
            output_dim=temporal_head_output_dim,
        )
        self.output_dim = temporal_head_output_dim * time_points

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("EEG input must have shape (batch, electrodes, time).")
        if x.size(1) != self.num_electrodes or x.size(2) != self.time_points:
            raise ValueError("The input shape does not match the DDNP configuration.")

        spatial_features = self.spatial_filter(self.gated_embedding(x))
        representations: list[Tensor] = []

        if self.use_self_attention:
            representations.append(self.temporal_transformer(self.positional_encoding(spatial_features)))
        if self.use_mstm:
            representations.append(self.multi_scale_temporal(x))
        if not representations:
            representations.append(spatial_features)

        fused_features = torch.stack(representations, dim=0).sum(dim=0)
        return self.temporal_head(fused_features).flatten(start_dim=1)
