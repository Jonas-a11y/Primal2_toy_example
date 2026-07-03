"""LMAPF task manager.

Wraps a GridWorld with:
  - Per-agent goals, reassigned on arrival with a min-distance constraint (LMAPF)
    or held fixed for the whole episode (one-shot).
  - Reward computation per Section IV.C:
      - -0.3 per timestep off-goal (per agent)
      - +5 on reaching goal (per agent)
      - -2 on collision (per agent involved)
  - Terminates after a fixed timestep budget, or (one-shot only) when every
    agent has reached its goal.

Paper metrics tracked for external reporting:
  - `goals_reached_count[i]`: number of arrivals for agent i (LMAPF throughput
    numerator).
  - `first_arrival_time[i]`: step at which agent i first reached its (initial)
    goal, or -1 if it never did. Used for makespan and success rate in one-shot.
  - `path_len[i]`: number of *non-stay*, *non-collided* moves agent i made
    (Section VI's "average path length").
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grid import GridWorld


@dataclass
class Config:
    size: int = 15
    density: float = 0.3
    typical_corridor_length: int = 5
    n_agents: int = 8
    episode_length: int = 256
    min_goal_dist: int = 3
    reward_off_goal: float = -0.3
    reward_on_goal: float = 5.0
    reward_collision: float = -2.0
    # One-shot mode: each agent has a single fixed goal; upon arrival the agent
    # "parks" on it (never moves again, contributes no more collisions). Episode
    # ends when all agents parked or timeout hit. Matches paper Section III.B.
    one_shot: bool = False


class LMAPFTask:
    def __init__(self, grid: GridWorld, config: Config, rng: np.random.Generator):
        self.grid = grid
        self.cfg = config
        self.rng = rng
        self.t = 0
        self.goals = np.zeros((grid.n_agents, 2), dtype=np.int32)
        self._empty_cells = np.argwhere(grid.obstacle_map == 0)
        self._assign_initial_goals()
        self.goals_reached_count = np.zeros(grid.n_agents, dtype=np.int32)
        # Paper metrics.
        self.first_arrival_time = np.full(grid.n_agents, -1, dtype=np.int32)
        self.path_len = np.zeros(grid.n_agents, dtype=np.int32)
        self.collision_count = np.zeros(grid.n_agents, dtype=np.int32)
        # One-shot bookkeeping: parked[i] == True means agent i has arrived and
        # stops taking part in the simulation.
        self.parked = np.zeros(grid.n_agents, dtype=bool)

    def _sample_far_cell(self, from_rc: tuple[int, int]) -> tuple[int, int]:
        """Sample an empty cell at least `min_goal_dist` (Euclidean) from `from_rc`."""
        for _ in range(200):
            idx = int(self.rng.integers(0, len(self._empty_cells)))
            r, c = int(self._empty_cells[idx, 0]), int(self._empty_cells[idx, 1])
            dr, dc = r - from_rc[0], c - from_rc[1]
            if (dr * dr + dc * dc) >= self.cfg.min_goal_dist ** 2:
                return (r, c)
        # Fallback: any empty cell.
        idx = int(self.rng.integers(0, len(self._empty_cells)))
        return (int(self._empty_cells[idx, 0]), int(self._empty_cells[idx, 1]))

    def _assign_initial_goals(self) -> None:
        for i in range(self.grid.n_agents):
            pos = (int(self.grid.positions[i, 0]), int(self.grid.positions[i, 1]))
            g = self._sample_far_cell(pos)
            self.goals[i, 0], self.goals[i, 1] = g

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Apply actions, reassign goals for agents that arrived, compute rewards.

        In one-shot mode, parked agents are forced to `stay` and their goals
        are never reassigned.

        Returns (rewards, arrived_flag, done).
        """
        # Force parked agents to stay so they neither move nor collide.
        actions = np.asarray(actions, dtype=np.int64).copy()
        if self.cfg.one_shot and self.parked.any():
            actions[self.parked] = 0

        prev_positions = self.grid.positions.copy()
        moved, collided = self.grid.step(actions)
        rewards = np.zeros(self.grid.n_agents, dtype=np.float32)
        arrived = np.zeros(self.grid.n_agents, dtype=bool)
        for i in range(self.grid.n_agents):
            if self.cfg.one_shot and self.parked[i]:
                # No reward, no metrics update; parked agents are out.
                continue
            if moved[i] and not collided[i]:
                self.path_len[i] += 1
            if collided[i]:
                self.collision_count[i] += 1
            pos = (int(self.grid.positions[i, 0]), int(self.grid.positions[i, 1]))
            g = (int(self.goals[i, 0]), int(self.goals[i, 1]))
            if pos == g:
                rewards[i] += self.cfg.reward_on_goal
                arrived[i] = True
                if self.first_arrival_time[i] < 0:
                    self.first_arrival_time[i] = self.t
                self.goals_reached_count[i] += 1
                if self.cfg.one_shot:
                    # Park; do not reassign.
                    self.parked[i] = True
                else:
                    new_g = self._sample_far_cell(pos)
                    self.goals[i, 0], self.goals[i, 1] = new_g
            else:
                rewards[i] += self.cfg.reward_off_goal
            if collided[i]:
                rewards[i] += self.cfg.reward_collision
        self.t += 1
        # Termination: timeout, or one-shot with everyone parked.
        done = self.t >= self.cfg.episode_length
        if self.cfg.one_shot and self.parked.all():
            done = True
        return rewards, arrived, done
