"""A3C + valid + BC loss components, matching PRIMAL2 Section IV.D exactly."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def discounted_returns(rewards: torch.Tensor, gamma: float, bootstrap: torch.Tensor) -> torch.Tensor:
    """Compute R_t = r_t + γ r_{t+1} + ... backward.

    rewards: (T,)
    bootstrap: scalar for R_{T+1} = V(o_{T+1}).
    Returns (T,).
    """
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    running = bootstrap
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def a3c_loss(
    logits: torch.Tensor,      # (T, A)
    values: torch.Tensor,      # (T,)
    actions: torch.Tensor,     # (T,) long
    rewards: torch.Tensor,     # (T,)
    valid_flags: torch.Tensor, # (T, A) 0/1 float
    bootstrap_value: torch.Tensor,  # scalar
    gamma: float = 0.95,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    actor_weight: float = 1.0,
    valid_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (total_loss, metrics)."""
    returns = discounted_returns(rewards, gamma, bootstrap_value.detach())
    advantages = returns - values.detach()
    # Actor: policy gradient with entropy bonus. Following the paper's Eq. 1:
    #   L_actor = mean( σ_H * H(π) - log π(a_t) * A_t )
    # We MAXIMIZE entropy, so add -σ_H*H to the loss.
    log_probs = F.log_softmax(logits, dim=-1)
    log_pa = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (T,)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
    actor_loss = -(log_pa * advantages).mean() - entropy_weight * entropy.mean()
    # Value loss: L2 (paper: "standard L2 loss L_value").
    value_loss = (returns - values).pow(2).mean()
    # Valid loss: per-action Bernoulli BCE on sigmoid(logits).
    # Paper's Eq. 3: L_valid = mean_t sum_i [log(v_i * σ(π_i)) + log((1-v_i)*(1-σ(π_i)))]
    # which is equivalent (up to sign) to standard BCE-with-logits. We use BCE
    # directly for numerical stability.
    valid_loss = F.binary_cross_entropy_with_logits(logits, valid_flags, reduction="mean")
    total = value_weight * value_loss + actor_weight * actor_loss + valid_weight * valid_loss
    return total, {
        "value_loss": float(value_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "valid_loss": float(valid_loss.item()),
        "entropy": float(entropy.mean().item()),
        "return_mean": float(returns.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
    }


def bc_loss(
    logits: torch.Tensor,           # (T, A)
    expert_actions: torch.Tensor,   # (T,) long
    valid_flags: torch.Tensor,      # (T, A) 0/1
    valid_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    bc = F.cross_entropy(logits, expert_actions, reduction="mean")
    valid_loss = F.binary_cross_entropy_with_logits(logits, valid_flags, reduction="mean")
    total = bc + valid_weight * valid_loss
    return total, {
        "bc_loss": float(bc.item()),
        "valid_loss": float(valid_loss.item()),
    }
