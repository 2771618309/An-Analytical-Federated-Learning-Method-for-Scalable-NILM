"""Feature extractor model for analytical NILM pretraining and AFL."""

import torch
import torch.nn as nn

from .layers import PatchEmbedding


class PatchResidualFeatureExtractor(nn.Module):
    """Patch residual feature extractor used as the shared NILM backbone.

    The layer order and feature extraction logic are kept aligned with the
    original implementation. Parameters that were fixed in the experiment code
    are exposed through the config files for the toy release.

    Args:
        num_classes: Number of output appliance classes for pretraining.
        input_length: Length of each one-dimensional waveform.
        patch_len: Number of waveform points per patch.
        patch_stride: Sliding-window stride used by patch embedding.
        patch_embed_dim: Embedding dimension for each waveform patch.
        out_channels: Number of channels in the residual convolution blocks.
        conv_out_channels: Number of output channels before flattening.
        hidden_dim: Hidden dimension of the pretraining classifier branch.
        dropout: Dropout probability used inside patch embedding.
    """

    def __init__(
        self,
        num_classes,
        input_length=150,
        patch_len=30,
        patch_stride=30,
        patch_embed_dim=150,
        out_channels=16,
        conv_out_channels=10,
        hidden_dim=512,
        dropout=0.0,
    ):
        """Initialize the patch residual feature extractor.

        Args:
            num_classes: Number of output appliance classes for pretraining.
            input_length: Length of each one-dimensional waveform.
            patch_len: Number of waveform points per patch.
            patch_stride: Sliding-window stride used by patch embedding.
            patch_embed_dim: Embedding dimension for each waveform patch.
            out_channels: Number of channels in the residual convolution blocks.
            conv_out_channels: Number of output channels before flattening.
            hidden_dim: Hidden dimension of the pretraining classifier branch.
            dropout: Dropout probability used inside patch embedding.
        """

        super().__init__()
        self.num_classes = num_classes
        self.input_length = input_length
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_embed_dim = patch_embed_dim
        self.out_channels = out_channels
        self.conv_out_channels = conv_out_channels
        self.feature_dim = conv_out_channels * patch_embed_dim

        patch_count = self._num_patches(input_length, patch_len, patch_stride)
        self.patch_embedding = PatchEmbedding(
            d_model=patch_embed_dim,
            patch_len=patch_len,
            stride=patch_stride,
            dropout=dropout,
        )
        self.conv1 = nn.Conv1d(
            in_channels=patch_count,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(patch_count)
        self.relu = nn.ReLU()
        self.resblock1 = self._make_resblock(out_channels)
        self.resblock2 = self._make_resblock(out_channels)
        self.resblock3 = self._make_resblock(out_channels)
        self.flatten = nn.Flatten()
        self.cov_out = nn.Conv1d(
            in_channels=out_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(conv_out_channels)
        self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    @staticmethod
    def _num_patches(input_length, patch_len, stride):
        """Compute the number of patches after right-side replication padding.

        Args:
            input_length: Length of the original waveform.
            patch_len: Number of waveform points in each patch.
            stride: Sliding-window stride.

        Returns:
            Number of extracted patches.
        """

        padded_length = input_length + stride - 1
        return ((padded_length - patch_len) // stride) + 1

    @staticmethod
    def _make_resblock(channels):
        """Build one residual 1D convolution block.

        Args:
            channels: Number of input and output channels.

        Returns:
            A sequential block with Conv1d, BatchNorm1d, ReLU, Conv1d, and
            BatchNorm1d layers.
        """

        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def _init_weights(self):
        """Initialize convolution, linear, and batch-normalization layers.

        Returns:
            None.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_features(self, x):
        """Extract waveform representations before the classifier branch.

        Args:
            x: Input waveform tensor with shape [batch_size, input_length].

        Returns:
            Feature tensor with shape [batch_size, feature_dim].
        """

        x = self.patch_embedding(x.unsqueeze(1))
        x = self.bn3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        identity = x
        out = self.resblock1(x)
        x = self.relu(out + identity)

        identity = x
        out = self.resblock2(x)
        x = self.relu(out + identity)

        identity = x
        out = self.resblock3(x)
        x = self.relu(out + identity)

        x = self.cov_out(x)
        x = self.bn2(x)
        return self.flatten(x)

    def forward(self, x, train=True):
        """Run feature extraction and optional pretraining classification.

        Args:
            x: Input waveform tensor with shape [batch_size, input_length].
            train: If True, return classifier logits and features. If False,
                return features twice for analytical-head compatibility.

        Returns:
            If train is True, returns (logits, features). If train is False,
            returns (features, features).
        """

        features = self.extract_features(x)
        if not train:
            return features, features
        hidden = self.fc1(features)
        hidden = self.relu(hidden)
        logits = self.fc2(hidden)
        return logits, features
