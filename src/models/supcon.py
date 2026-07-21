"""Supervised Contrastive (SupCon) components.

Contains:
- SupConModel: wraps LSTMEncoder + ProjectionHead for pretraining.
- SupConLoss: the supervised contrastive loss itself
  (Khosla et al., 2020 - https://arxiv.org/abs/2004.11362).
"""
import torch
from torch import nn

from .encoder import LSTMEncoder
from .projection_head import ProjectionHead


class SupConModel(nn.Module):
    """Encoder + projection head, used only for contrastive pretraining."""

    def __init__(self, input_size: int, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_dim)
        self.projection_head = ProjectionHead(hidden_dim, projection_dim)

    def forward(self, x):
        return self.projection_head(self.encoder(x))

    def get_features(self, x):
        """Raw encoder features (no projection) - what the downstream
        classifier is trained on after pretraining."""
        with torch.no_grad():
            return self.encoder(x)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Adapted from the reference implementation by Yonglong Tian
    (https://github.com/HobbitLong/SupContrast).
    """

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all",
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [batch_size, n_views, feature_dim] (already L2-normalized)
            labels:   [batch_size] ground-truth class labels (optional)
            mask:     precomputed contrastive mask (optional, mutually
                      exclusive with `labels`)
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...], "
                              "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
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
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask to [anchor_count * bsz, contrast_count * bsz]
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # compute log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-12)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
