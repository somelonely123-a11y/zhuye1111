"""Template-guided analytical path of DP-Net."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .aspm import AnalyticalScorePreservationModule


def default_target_frequencies(num_classes: int = 40) -> list[float]:
    """Return the Benchmark/BETA target-frequency sequence."""
    return [8.0 + 0.2 * class_index for class_index in range(num_classes)]


def build_standard_harmonic_templates(
    frequencies: Sequence[float],
    time_points: int,
    sampling_rate: int,
    harmonic_orders: int = 5,
) -> Tensor:
    """Construct sine-cosine reference templates for all target frequencies."""
    if harmonic_orders < 1:
        raise ValueError("harmonic_orders must be at least 1.")

    time = torch.arange(time_points, dtype=torch.float32) / sampling_rate
    templates = torch.empty(len(frequencies), 2 * harmonic_orders, time_points, dtype=torch.float32)
    for class_index, frequency in enumerate(frequencies):
        for harmonic in range(1, harmonic_orders + 1):
            templates[class_index, 2 * (harmonic - 1)] = torch.sin(2.0 * torch.pi * harmonic * frequency * time)
            templates[class_index, 2 * (harmonic - 1) + 1] = torch.cos(2.0 * torch.pi * harmonic * frequency * time)
    return templates


def build_empirical_templates(
    source_eeg: Tensor,
    source_labels: Tensor,
    frequencies: Sequence[float],
    sampling_rate: int,
) -> Tensor:
    """Build periodic source-domain empirical templates from training trials only."""
    source_eeg = torch.as_tensor(source_eeg, dtype=torch.float32)
    source_labels = torch.as_tensor(source_labels, dtype=torch.long).reshape(-1)

    if source_eeg.ndim != 3:
        raise ValueError("source_eeg must have shape (trials, electrodes, time).")
    if source_eeg.size(0) != source_labels.numel():
        raise ValueError("source_eeg and source_labels must contain the same number of trials.")

    _, num_electrodes, time_points = source_eeg.shape
    templates = source_eeg.new_empty(len(frequencies), num_electrodes, time_points)

    for class_index, frequency in enumerate(frequencies):
        class_trials = source_eeg[source_labels == class_index]
        if class_trials.numel() == 0:
            raise ValueError(f"No source-domain training trial is available for class {class_index}.")

        cycle_length = max(1, int(round(sampling_rate / float(frequency))))
        cycle_count = time_points // cycle_length

        if cycle_count == 0:
            templates[class_index] = class_trials.mean(dim=0)
            continue

        valid_length = cycle_count * cycle_length
        cycles = class_trials[:, :, :valid_length].reshape(
            class_trials.size(0),
            num_electrodes,
            cycle_count,
            cycle_length,
        )
        single_cycle = cycles.mean(dim=2)
        repeats = (time_points + cycle_length - 1) // cycle_length
        reconstructed = single_cycle.repeat(1, 1, repeats)[:, :, :time_points]
        templates[class_index] = reconstructed.mean(dim=0)

    return templates


class TemplateGuidedAnalyticalPath(nn.Module):
    """TGAP: dual-template QR/SVD similarity features with temporal shifts."""

    def __init__(
        self,
        num_classes: int,
        time_points: int,
        sampling_rate: int = 250,
        frequencies: Sequence[float] | None = None,
        harmonic_orders: int = 5,
        max_shift_samples: int = 6,
        shift_penalty_sigma: float = 6.0,
    ) -> None:
        super().__init__()
        if max_shift_samples < 0:
            raise ValueError("max_shift_samples must be non-negative.")
        if shift_penalty_sigma <= 0:
            raise ValueError("shift_penalty_sigma must be positive.")

        self.num_classes = num_classes
        self.time_points = time_points
        self.sampling_rate = sampling_rate
        self.frequencies = list(frequencies) if frequencies is not None else default_target_frequencies(num_classes)
        if len(self.frequencies) != num_classes:
            raise ValueError("The number of target frequencies must equal num_classes.")

        shifts = torch.arange(-max_shift_samples, max_shift_samples + 1, dtype=torch.long)
        penalties = torch.exp(-(shifts.to(torch.float32) ** 2) / (2.0 * shift_penalty_sigma**2))
        self.center_shift_index = max_shift_samples
        self.register_buffer("shifts", shifts, persistent=True)
        self.register_buffer("shift_penalties", penalties / penalties.max(), persistent=True)

        standard_templates = build_standard_harmonic_templates(
            self.frequencies,
            time_points,
            sampling_rate,
            harmonic_orders,
        )
        self.register_buffer("standard_templates", standard_templates, persistent=True)
        self.register_buffer("empirical_templates", torch.empty(0), persistent=True)
        self.register_buffer("standard_q", self._template_qr(standard_templates), persistent=True)
        self.register_buffer("empirical_q", torch.empty(0), persistent=True)
        self.aspm = AnalyticalScorePreservationModule()

    def _template_qr(self, templates: Tensor) -> Tensor:
        shifted_templates = torch.stack(
            [torch.roll(templates, shifts=int(shift.item()), dims=-1) for shift in self.shifts],
            dim=0,
        )
        matrices = shifted_templates.permute(0, 1, 3, 2)
        matrices = matrices - matrices.mean(dim=2, keepdim=True)
        return torch.linalg.qr(matrices, mode="reduced").Q

    @torch.no_grad()
    def set_empirical_templates(self, templates: Tensor) -> None:
        """Register empirical templates built from a source-domain training subset."""
        templates = torch.as_tensor(
            templates,
            dtype=self.standard_templates.dtype,
            device=self.standard_templates.device,
        )
        if templates.ndim != 3:
            raise ValueError(
                "Empirical templates must have shape (num_classes, electrodes, time_points)."
            )
        if templates.size(0) != self.num_classes or templates.size(2) != self.time_points:
            raise ValueError(
                "Empirical templates must have shape (num_classes, electrodes, time_points)."
            )

        self.empirical_templates = templates.contiguous()
        self.empirical_q = self._template_qr(self.empirical_templates)

    def _score_templates(self, eeg_q: Tensor, template_q: Tensor) -> tuple[Tensor, Tensor]:
        interaction = torch.einsum("btc,sktd->bskcd", eeg_q, template_q)
        singular_values = torch.linalg.svdvals(interaction)
        shift_scores = singular_values[..., 0].clamp(min=0.0, max=1.0)
        center_scores = shift_scores[:, self.center_shift_index, :]
        penalized_scores = (shift_scores * self.shift_penalties.view(1, -1, 1)).amax(dim=1)
        return center_scores, penalized_scores

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError("EEG input must have shape (batch, electrodes, time).")
        if x.size(2) != self.time_points:
            raise ValueError("The input time length does not match the TGAP configuration.")
        if self.empirical_q.numel() == 0:
            raise RuntimeError(
                "Empirical templates have not been set. Build them from the source-domain training subset before inference."
            )

        centered_eeg = x - x.mean(dim=-1, keepdim=True)
        eeg_q = torch.linalg.qr(centered_eeg.transpose(1, 2), mode="reduced").Q
        standard_center, standard_shifted = self._score_templates(eeg_q, self.standard_q)
        empirical_center, empirical_shifted = self._score_templates(eeg_q, self.empirical_q)
        return self.aspm(
            standard_center,
            empirical_center,
            standard_shifted,
            empirical_shifted,
        )
