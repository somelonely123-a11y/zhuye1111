"""Reusable layers for the data-driven neural path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GatedEmbedding(nn.Module):
    """Gated embedding block for multichannel EEG input."""

    def __init__(self, num_electrodes: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.num_electrodes = num_electrodes
        self.hidden_dim = hidden_dim
        self.fusion = nn.Conv2d(
            in_channels=1 + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False,
        )
        self.gates = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=1,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.fusion.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("EEG input must have shape (batch, electrodes, time).")
        if x.size(1) != self.num_electrodes:
            raise ValueError("The input electrode count does not match the model configuration.")

        x = x.unsqueeze(1)
        zeros = x.new_zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3))
        embedded = self.fusion(torch.cat((x, zeros), dim=1))

        content, input_gate, output_gate = self.gates(embedded).chunk(3, dim=1)
        return torch.tanh(content) * torch.sigmoid(input_gate) * torch.sigmoid(output_gate)


class CrossElectrodeSpatialFilter(nn.Module):
    """Spatial convolution spanning all electrodes without temporal pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_electrodes: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(num_electrodes, 1),
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Input must have shape (batch, channels, electrodes, time).")
        if x.size(1) != self.in_channels or x.size(2) != self.num_electrodes:
            raise ValueError("The spatial-filter input shape does not match the model configuration.")
        return self.conv(x).squeeze(2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal tokens."""

    def __init__(self, embedding_dim: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_length, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding.transpose(0, 1).unsqueeze(0), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, features, time).")
        if x.size(2) > self.encoding.size(2):
            raise ValueError("Input time length exceeds the configured positional-encoding length.")
        return x + self.encoding[:, :, : x.size(2)]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention applied over the temporal axis."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, time, features).")

        batch_size, time_points, embedding_dim = x.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError("The feature dimension does not match the attention configuration.")

        query = self.query(x).view(batch_size, time_points, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, time_points, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, time_points, self.num_heads, self.head_dim).transpose(1, 2)

        weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        weights = self.dropout(F.softmax(weights, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, time_points, embedding_dim)
        return self.output(attended)


class TemporalFeedForward(nn.Module):
    """Position-wise feed-forward network for temporal features."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TemporalTransformerBlock(nn.Module):
    """One preconfigured Transformer block for temporal modeling."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.feedforward = TemporalFeedForward(embedding_dim, feedforward_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, features, time).")
        tokens = x.transpose(1, 2)
        tokens = self.norm1(tokens + self.dropout(self.attention(tokens)))
        tokens = self.norm2(tokens + self.dropout(self.feedforward(tokens)))
        return tokens.transpose(1, 2)


class MultiScaleTemporalModule(nn.Module):
    """Parallel temporal convolutions followed by channel fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (31, 15, 7, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty.")
        if out_channels % len(kernel_sizes) != 0:
            raise ValueError("out_channels must be divisible by the number of temporal branches.")

        branch_channels = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.fusion = nn.Conv1d(branch_channels * len(kernel_sizes), out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, electrodes, time).")
        features = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(features, dim=1))


class TemporalFeatureHead(nn.Module):
    """Time-step-wise feature compression from the DDNP output."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, features, time).")
        return self.network(x.transpose(1, 2)).transpose(1, 2)
