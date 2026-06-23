"""Analytical score preservation module of DP-Net."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AnalyticalScorePreservationModule(nn.Module):
    """Preserve standard, dual-template, and shift-penalized score vectors."""

    def forward(
        self,
        standard_center_scores: Tensor,
        empirical_center_scores: Tensor,
        standard_shift_scores: Tensor,
        empirical_shift_scores: Tensor,
    ) -> Tensor:
        tensors = (
            standard_center_scores,
            empirical_center_scores,
            standard_shift_scores,
            empirical_shift_scores,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("All ASPM inputs must have shape (batch, classes).")

        dual_template_scores = torch.maximum(standard_center_scores, empirical_center_scores)
        shift_penalized_scores = torch.maximum(standard_shift_scores, empirical_shift_scores)
        return torch.cat(
            (standard_center_scores, dual_template_scores, shift_penalized_scores),
            dim=1,
        )
