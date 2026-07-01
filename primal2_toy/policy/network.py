"""PRIMAL/PRIMAL2 policy network.

Architecture faithfully reproduces Figure 3 of PRIMAL2 (Section IV.D):

  spatial obs (C, F, F) -> VGG block 1 (64 ch) -> VGG block 2 (128 ch) -> 1x1 conv -> flatten -> FC
  scalars (3,) -> FC
  concat -> FC -> FC   (residual shortcut here)
       -> LSTM         (residual add on LSTM output)
       -> policy head (5-way softmax logits)
       -> value head  (scalar)

Because VGG-style blocks with two MaxPool2d(2) halve the spatial dims twice,
an 11x11 FOV would reduce to 2x2 after the second pool (11 -> 5 -> 2). That
still leaves the 1x1 conv well-defined. We do not use adaptive pooling to keep
the model 1:1 with the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _vgg_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class PolicyNet(nn.Module):
    def __init__(
        self,
        n_spatial_channels: int = 11,
        n_scalars: int = 3,
        n_actions: int = 5,
        fov: int = 11,
        conv1_ch: int = 64,
        conv2_ch: int = 128,
        conv3_ch: int = 32,
        fc_dim: int = 512,
        lstm_dim: int = 512,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.lstm_dim = lstm_dim

        self.vgg1 = _vgg_block(n_spatial_channels, conv1_ch)
        self.vgg2 = _vgg_block(conv1_ch, conv2_ch)
        # 1x1 conv folding channels down.
        self.conv3 = nn.Conv2d(conv2_ch, conv3_ch, kernel_size=1)
        # Compute flattened size after two 2x2 pools.
        pooled = fov // 2 // 2  # for fov=11 -> 2
        self.spatial_flat = conv3_ch * pooled * pooled
        self.fc_spatial = nn.Linear(self.spatial_flat, fc_dim)
        # Scalar branch.
        self.fc_scalar = nn.Linear(n_scalars, 12)
        # Two FC layers after concat.
        self.fc_concat1 = nn.Linear(fc_dim + 12, fc_dim)
        self.fc_concat2 = nn.Linear(fc_dim, fc_dim)
        # LSTM.
        self.lstm = nn.LSTMCell(fc_dim, lstm_dim)
        # Residual projection (fc_dim -> lstm_dim if different).
        self.residual_proj = nn.Linear(fc_dim, lstm_dim) if fc_dim != lstm_dim else nn.Identity()
        # Heads.
        self.policy_head = nn.Linear(lstm_dim, n_actions)
        self.value_head = nn.Linear(lstm_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.lstm_dim, device=device),
            torch.zeros(batch_size, self.lstm_dim, device=device),
        )

    def forward(
        self,
        spatial: torch.Tensor,  # (B, C, F, F)
        scalars: torch.Tensor,  # (B, 3)
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.vgg1(spatial)
        x = self.vgg2(x)
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc_spatial(x))
        s = F.relu(self.fc_scalar(scalars))
        z = torch.cat([x, s], dim=1)
        z = F.relu(self.fc_concat1(z))
        z_pre_lstm = F.relu(self.fc_concat2(z))
        # LSTM step.
        h, c = self.lstm(z_pre_lstm, hidden)
        # Residual shortcut from concat output to LSTM output.
        h = h + self.residual_proj(z_pre_lstm)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value, (h, c)
