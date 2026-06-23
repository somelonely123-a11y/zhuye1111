"""DP-Net model definition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ddnp import DataDrivenNeuralPath
from .tgap import TemplateGuidedAnalyticalPath


class DPNet(nn.Module):
    """Dual-path network for zero-calibration cross-subject SSVEP decoding."""

    def __init__(
        self,
        num_classes: int = 40,
        num_electrodes: int = 9,
        time_points: int = 150,
        sampling_rate: int = 250,
        frequencies: Sequence[float] | None = None,
        gated_embedding_dim: int = 32,
        spatial_dim: int = 64,
        temporal_kernel_sizes: tuple[int, ...] = (31, 15, 7, 1),
        attention_heads: int = 8,
        transformer_feedforward_dim: int = 256,
        transformer_dropout: float = 0.0,
        temporal_dropout: float = 0.0,
        temporal_head_hidden_dim: int = 16,
        temporal_head_output_dim: int = 4,
        harmonic_orders: int = 5,
        max_shift_samples: int = 6,
        shift_penalty_sigma: float = 6.0,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.5,
        use_ddnp: bool = True,
        use_tgap: bool = True,
        use_mstm: bool = True,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if not use_ddnp and not use_tgap:
            raise ValueError("At least one path must be enabled.")

        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.time_points = time_points
        self.use_ddnp = use_ddnp
        self.use_tgap = use_tgap

        if use_ddnp:
            self.ddnp = DataDrivenNeuralPath(
                num_electrodes=num_electrodes,
                time_points=time_points,
                gated_embedding_dim=gated_embedding_dim,
                spatial_dim=spatial_dim,
                temporal_kernel_sizes=temporal_kernel_sizes,
                attention_heads=attention_heads,
                transformer_feedforward_dim=transformer_feedforward_dim,
                transformer_dropout=transformer_dropout,
                temporal_dropout=temporal_dropout,
                temporal_head_hidden_dim=temporal_head_hidden_dim,
                temporal_head_output_dim=temporal_head_output_dim,
                use_mstm=use_mstm,
                use_self_attention=use_self_attention,
            )
            ddnp_dim = self.ddnp.output_dim
        else:
            self.ddnp = None
            ddnp_dim = 0

        if use_tgap:
            self.tgap = TemplateGuidedAnalyticalPath(
                num_classes=num_classes,
                time_points=time_points,
                sampling_rate=sampling_rate,
                frequencies=frequencies,
                harmonic_orders=harmonic_orders,
                max_shift_samples=max_shift_samples,
                shift_penalty_sigma=shift_penalty_sigma,
            )
            tgap_dim = 3 * num_classes
        else:
            self.tgap = None
            tgap_dim = 0

        self.fused_feature_dim = ddnp_dim + tgap_dim
        self.feature_normalization = nn.BatchNorm1d(self.fused_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, classifier_hidden_dim),
            nn.ELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    @torch.no_grad()
    def set_empirical_templates(self, templates: Tensor) -> None:
        """Set fold-specific empirical templates for the analytical path."""
        if self.tgap is None:
            raise RuntimeError("The analytical path is disabled.")
        self.tgap.set_empirical_templates(templates)

    def _normalize_fused_features(self, features: Tensor) -> Tensor:
        if self.training and features.size(0) == 1:
            return F.batch_norm(
                features,
                self.feature_normalization.running_mean,
                self.feature_normalization.running_var,
                self.feature_normalization.weight,
                self.feature_normalization.bias,
                training=False,
                momentum=0.0,
                eps=self.feature_normalization.eps,
            )
        return self.feature_normalization(features)

    def forward(self, x: Tensor, return_features: bool = False):
        if x.ndim == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError("EEG input must have shape (batch, electrodes, time).")
        if x.size(1) != self.num_electrodes or x.size(2) != self.time_points:
            raise ValueError("The input shape does not match the DP-Net configuration.")

        feature_groups: dict[str, Tensor] = {}
        if self.ddnp is not None:
            feature_groups["ddnp"] = self.ddnp(x)
        if self.tgap is not None:
            feature_groups["tgap"] = self.tgap(x)

        fused_features = torch.cat(tuple(feature_groups.values()), dim=1)
        logits = self.classifier(self._normalize_fused_features(fused_features))

        if return_features:
            feature_groups["fused"] = fused_features
            return logits, feature_groups
        return logits
