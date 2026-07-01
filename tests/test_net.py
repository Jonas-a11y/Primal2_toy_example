"""Smoke test for PolicyNet and losses."""
import numpy as np
import torch
from primal2_toy.policy.network import PolicyNet
from primal2_toy.policy.losses import a3c_loss, bc_loss


def main():
    torch.manual_seed(0)
    net = PolicyNet(n_spatial_channels=11, n_scalars=3, n_actions=5, fov=11)
    B, C, F = 4, 11, 11
    spatial = torch.randn(B, C, F, F)
    scalars = torch.randn(B, 3)
    h, c = net.init_hidden(B, torch.device("cpu"))
    logits, values, (h, c) = net(spatial, scalars, (h, c))
    assert logits.shape == (B, 5)
    assert values.shape == (B,)

    # A3C loss.
    T = 6
    L = torch.randn(T, 5, requires_grad=True)
    V = torch.randn(T, requires_grad=True)
    A = torch.randint(0, 5, (T,))
    R = torch.randn(T)
    valid = (torch.rand(T, 5) > 0.5).float()
    bootstrap = torch.tensor(0.0)
    loss, m = a3c_loss(L, V, A, R, valid, bootstrap)
    loss.backward()
    assert L.grad is not None
    print("a3c metrics:", m)

    # BC loss.
    L2 = torch.randn(T, 5, requires_grad=True)
    A2 = torch.randint(0, 5, (T,))
    valid2 = (torch.rand(T, 5) > 0.5).float()
    loss2, m2 = bc_loss(L2, A2, valid2)
    loss2.backward()
    print("bc metrics:", m2)
    print("net_smoke OK")


if __name__ == "__main__":
    main()
