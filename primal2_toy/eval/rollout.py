"""Rollout a trained policy on a seeded scenario."""
from __future__ import annotations

import numpy as np
import torch

from ..env.grid import GridWorld
from ..env.maze import generate_maze
from ..env.corridor import analyze
from ..env.lmapf import LMAPFTask, Config as LMAPFConfig
from ..obs.builder import ObservationBuilder, ObsSpec
from ..policy.network import PolicyNet
from ..train.validity import compute_valid_actions
from ..train.trainer import _batch_obs


def build_scenario(size: int, density: float, corridor_length: int, n_agents: int, seed: int):
    rng = np.random.default_rng(seed)
    m = generate_maze(size, density, corridor_length, rng)
    empties = np.argwhere(m == 0)
    n_agents = min(n_agents, len(empties))
    idx = rng.choice(len(empties), size=n_agents, replace=False)
    starts = [tuple(empties[i]) for i in idx]
    grid = GridWorld(m, starts)
    lmapf_cfg = LMAPFConfig(
        size=size, density=density, typical_corridor_length=corridor_length,
        n_agents=n_agents, episode_length=1_000_000,  # effectively no timeout
    )
    task = LMAPFTask(grid, lmapf_cfg, rng)
    corridors, cell_to_id = analyze(m)
    return grid, task, corridors, cell_to_id


class Rollout:
    """Iterator over environment steps under a trained policy."""

    def __init__(
        self,
        net: PolicyNet,
        grid: GridWorld,
        task: LMAPFTask,
        corridors,
        cell_to_id: np.ndarray,
        device: torch.device,
        obs_spec: ObsSpec | None = None,
        greedy: bool = True,
    ):
        self.net = net.eval()
        self.grid = grid
        self.task = task
        self.corridors = corridors
        self.cell_to_id = cell_to_id
        self.device = device
        self.obs_spec = obs_spec or ObsSpec(fov=11, n_pred=3)
        self.builder = ObservationBuilder(grid, task.goals, corridors, cell_to_id, self.obs_spec)
        self.hidden = net.init_hidden(grid.n_agents, device)
        self.greedy = greedy
        self.step_idx = 0
        self.total_collisions = 0
        self.total_arrivals = 0

    @torch.no_grad()
    def step(self) -> dict:
        sp, sc = _batch_obs(self.builder, self.grid.n_agents)
        sp = sp.to(self.device)
        sc = sc.to(self.device)
        logits, values, self.hidden = self.net(sp, sc, self.hidden)
        valid = compute_valid_actions(self.grid, self.corridors, self.cell_to_id)
        valid_t = torch.from_numpy(valid).to(self.device)
        masked = logits.clone()
        masked[valid_t == 0] = -1e9
        if self.greedy:
            actions = masked.argmax(dim=-1).cpu().numpy().astype(np.int64)
        else:
            probs = torch.softmax(masked, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy().astype(np.int64)
        rewards, arrived, done = self.task.step(actions)
        self.total_arrivals += int(arrived.sum())
        self.step_idx += 1
        return {
            "actions": actions,
            "arrived": arrived,
            "positions": self.grid.positions.copy(),
            "goals": self.task.goals.copy(),
            "step": self.step_idx,
        }
