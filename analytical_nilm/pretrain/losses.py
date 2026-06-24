"""Loss functions used by the toy pretraining pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss for appliance class prediction.

    Args:
        logits: Class logits with shape [batch_size, num_classes].
        labels: Integer class labels with shape [batch_size].

    Returns:
        Scalar cross-entropy loss tensor.
    """

    return F.cross_entropy(logits, labels)


class SupConLoss(nn.Module):
    """Supervised contrastive loss used for feature pretraining.

    Args:
        temperature: Temperature used to scale pairwise similarities.
        contrast_mode: Anchor selection mode. "all" uses all views as anchors;
            "one" uses only the first view.
        base_temperature: Normalization temperature for the final loss value.
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        """Initialize supervised contrastive loss hyperparameters.

        Args:
            temperature: Temperature used to scale pairwise similarities.
            contrast_mode: Anchor selection mode. "all" uses all views as
                anchors; "one" uses only the first view.
            base_temperature: Normalization temperature for the final loss.
        """

        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute supervised contrastive loss.

        Args:
            features: Feature tensor with shape [batch_size, n_views, feature_dim].
                Extra trailing dimensions are flattened automatically.
            labels: Optional integer labels with shape [batch_size]. Samples
                with the same label are treated as positives.
            mask: Optional binary positive-pair mask with shape
                [batch_size, batch_size]. Cannot be provided together with labels.

        Returns:
            Scalar supervised contrastive loss tensor.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("features must have shape [batch_size, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("labels and mask cannot both be provided")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("label count does not match feature count")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"unknown contrast_mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
