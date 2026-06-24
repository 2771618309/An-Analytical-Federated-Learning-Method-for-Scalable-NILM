"""Patch embedding layers used by the NILM feature extractor."""

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding used by the patch embedding layer.

    Args:
        d_model: Embedding dimension for each patch token.
        max_len: Maximum supported token sequence length.
    """

    def __init__(self, d_model, max_len=5000):
        """Initialize a fixed sinusoidal positional encoding table.

        Args:
            d_model: Embedding dimension for each patch token.
            max_len: Maximum supported token sequence length.
        """

        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Return positional encodings matching the input sequence length.

        Args:
            x: Tensor with shape [batch_size, sequence_length, d_model].

        Returns:
            Positional encoding tensor with shape [1, sequence_length, d_model].
        """

        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Token embedding layer implemented with 1D convolution.

    Args:
        c_in: Number of input channels or patch values.
        d_model: Output embedding dimension.
    """

    def __init__(self, c_in, d_model):
        """Initialize the convolutional token embedding.

        Args:
            c_in: Number of input channels or patch values.
            d_model: Output embedding dimension.
        """

        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        """Embed patch tokens with circular 1D convolution.

        Args:
            x: Tensor with shape [batch_size, sequence_length, c_in].

        Returns:
            Embedded tensor with shape [batch_size, sequence_length, d_model].
        """

        x = self.tokenConv(x.permute(0, 2, 1))
        return x.transpose(1, 2)


class PatchEmbedding(nn.Module):
    """Patch embedding layer for one-dimensional NILM waveforms.

    Args:
        d_model: Patch token embedding dimension.
        patch_len: Number of waveform points in each patch.
        stride: Sliding-window stride used to extract patches.
        dropout: Dropout probability after value and position embedding.
    """

    def __init__(self, d_model, patch_len, stride, dropout):
        """Initialize patch extraction, token embedding, and dropout layers.

        Args:
            d_model: Patch token embedding dimension.
            patch_len: Number of waveform points in each patch.
            stride: Sliding-window stride used to extract patches.
            dropout: Dropout probability after value and position embedding.
        """

        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride - 1))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Convert a waveform batch into embedded patch tokens.

        Args:
            x: Tensor with shape [batch_size, channels, input_length].

        Returns:
            Embedded patch tensor with shape
            [batch_size * channels, num_patches, d_model].
        """

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
