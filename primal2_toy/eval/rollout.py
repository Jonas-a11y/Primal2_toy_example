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


def build_scenario(
    size: int,
    density: float,
    corridor_length: int,
    n_agents: int,
    seed: int,
    *,
    episode_length: int = 1_000_000,
    one_shot: bool = False,
):
    """Build a fresh env from (size, density, corridor_length, n_agents, seed).

    Parameters
    ----------
    episode_length : hard timeout on the task, in steps.
    one_shot : if True, each agent has one fixed goal and parks on arrival
        (paper's one-shot MAPF).
    """
    rng = np.random.default_rng(seed)
    m = generate_maze(size, density, corridor_length, rng)
    empties = np.argwhere(m == 0)
    n_agents = min(n_agents, len(empties))
    idx = rng.choice(len(empties), size=n_agents, replace=False)
    starts = [tuple(empties[i]) for i in idx]
    grid = GridWorld(m, starts)
    lmapf_cfg = LMAPFConfig(
        size=size, density=density, typical_corridor_length=corridor_length,
        n_agents=n_agents, episode_length=episode_length, one_shot=one_shot,
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
        self.done = False

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
        self.done = bool(done)
        return {
            "actions": actions,
            "arrived": arrived,
            "positions": self.grid.positions.copy(),
            "goals": self.task.goals.copy(),
            "step": self.step_idx,
            "done": self.done,
        }

    # ---- Paper metrics ----
    def paper_metrics(self, agent_subset: np.ndarray | None = None) -> dict:
        """Compute the metrics reported in Section VI of the paper.

        agent_subset: optional boolean mask over agents. Restricts metrics to
        that subset (used for the 95 % success rate variant).

        For LMAPF the relevant metric is `throughput`; the others are all
        one-shot metrics but they're always computed so callers can pick.
        """
        n = self.grid.n_agents
        mask = np.ones(n, dtype=bool) if agent_subset is None else agent_subset
        n_sub = int(mask.sum())
        arrivals_bool = self.task.first_arrival_time >= 0
        arrived_sub = arrivals_bool & mask
        success = bool(arrived_sub.sum() == n_sub) if n_sub else True
        # Makespan: max over subset of first-arrival time (or -1 if any didn't arrive).
        if success:
            makespan = int(self.task.first_arrival_time[mask].max())
        else:
            makespan = -1
        # Average path length over subset agents that arrived.
        arrived_lens = self.task.path_len[arrived_sub]
        avg_path_len = float(arrived_lens.mean()) if len(arrived_lens) else 0.0
        # Throughput = total (repeated) arrivals across all agents / step_idx.
        # In LMAPF the paper defines throughput per-timestep of the whole episode
        # (Section III.C); here we use `self.step_idx` which equals the effective
        # episode length.
        throughput = self.total_arrivals / max(1, self.step_idx)
        collision_rate = float(self.task.collision_count.sum()) / max(1, self.step_idx * n)
        return {
            "success_100": success,  # True iff every subset agent reached its goal
            "n_arrived": int(arrived_sub.sum()),
            "n_subset": n_sub,
            "makespan": makespan,
            "avg_path_len": avg_path_len,
            "throughput": throughput,
            "total_arrivals": int(self.total_arrivals),
            "steps": int(self.step_idx),
            "collision_rate_per_agent_step": collision_rate,
        }
