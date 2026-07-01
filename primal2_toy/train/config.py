"""Training configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Environment randomization ranges (paper V.B.2: size 10-70, density 0.2-0.7,
    # corridor 3-21). We shrink to fit the training budget.
    size_choices: tuple[int, ...] = (10, 15, 20)
    density_low: float = 0.2
    density_high: float = 0.5
    corridor_choices: tuple[int, ...] = (1, 3, 5, 7)
    n_agents: int = 8

    # Episode lengths (paper V.B.1).
    rl_episode_length: int = 256
    il_episode_length: int = 64

    # Loss weights (paper's Section IV.D says "manually tuned").
    # Paper defaults: α=0.5, β=1.0, ζ=0.5, σ_H=0.01. We bumped entropy weight for
    # our single-worker regime (paper's 9 workers provide diversity we lack).
    value_weight: float = 0.5
    actor_weight: float = 1.0
    valid_weight: float = 0.5
    entropy_weight: float = 0.05
    gamma: float = 0.95

    # Optimizer (paper V.B.1). We also bump the initial LR modestly since our
    # single-worker training sees fewer diverse rollouts per unit compute.
    lr: float = 5e-5
    lr_decay_ref_episode: int = 1000  # decay proportionally to 1/sqrt(episode / ref)
    grad_clip: float = 5.0

    # Ratio & schedule.
    il_prob: float = 0.5
    # IL warmup: for the first N episodes force IL-only, so the network gets a
    # decent policy initialization before RL episodes begin. Paper doesn't
    # explicitly do this; we add it because with only one worker the RL signal
    # early on is very noisy and can destabilize the policy.
    il_warmup_episodes: int = 500

    # Observation.
    fov: int = 11
    n_pred: int = 3

    # Training scale.
    total_episodes: int = 30_000
    num_workers: int = 4

    # Checkpointing / logging.
    ckpt_every_episodes: int = 500
    log_every_episodes: int = 20

    # Device: "auto" resolves to mps -> cuda -> cpu.
    device: str = "auto"

    # Paths.
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
