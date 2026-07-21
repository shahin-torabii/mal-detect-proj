"""MLP projection head used only during SupCon pretraining.

Maps encoder features into the (L2-normalized) space where the
supervised contrastive loss is computed. It is discarded after
pretraining — the downstream classifier is trained on the encoder's
raw features, not on the projected ones.
"""
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)
