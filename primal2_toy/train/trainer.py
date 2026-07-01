"""Single-process training loop.

Strategy: we implement a *sequential* training loop first (one worker) that
performs both RL and IL episodes and accumulates gradients. This is easier to
debug and produces valid checkpoints. Multi-worker A3C-style parallelism can be
layered on afterward via torch.multiprocessing.

Per iteration:
  1. Randomize env (size, density, corridor length).
  2. With prob 0.5, run an IL episode; else run an RL episode.
  3. Accumulate loss over agents, backward, step optimizer.
  4. Occasionally checkpoint and log.
"""
from __future__ import annotations

import os
import time
import csv
import math
import numpy as np
import torch
from typing import Any

from ..env.grid import GridWorld
from ..env.maze import generate_maze
from ..env.corridor import analyze
from ..env.lmapf import LMAPFTask, Config as LMAPFConfig
from ..obs.builder import ObservationBuilder, ObsSpec
from ..policy.network import PolicyNet
from ..policy.losses import a3c_loss, bc_loss
from ..expert.prioritized_astar import plan as expert_plan
from .config import TrainConfig
from .validity import compute_valid_actions


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _random_env(cfg: TrainConfig, rng: np.random.Generator) -> tuple[GridWorld, LMAPFTask, list[Any], np.ndarray]:
    size = int(rng.choice(cfg.size_choices))
    density = float(rng.uniform(cfg.density_low, cfg.density_high))
    corridor_len = int(rng.choice(cfg.corridor_choices))
    n = cfg.n_agents
    # Cap n to what fits on this map (leave at least 4 free cells).
    max_agents = max(1, (size * size) // 4)
    n = min(n, max_agents)

    m = generate_maze(size, density, corridor_len, rng)
    empties = np.argwhere(m == 0)
    idx = rng.choice(len(empties), size=n, replace=False)
    starts = [tuple(empties[i]) for i in idx]
    grid = GridWorld(m, starts)
    lmapf_cfg = LMAPFConfig(
        size=size,
        density=density,
        typical_corridor_length=corridor_len,
        n_agents=n,
        episode_length=cfg.rl_episode_length,  # overridden per episode-type
    )
    task = LMAPFTask(grid, lmapf_cfg, rng)
    corridors, cell_to_id = analyze(m)
    return grid, task, corridors, cell_to_id


def _batch_obs(builder: ObservationBuilder, n_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
    builder.refresh_step_cache()
    spatials, scalars = [], []
    for i in range(n_agents):
        s, sc = builder.build(i)
        spatials.append(s)
        scalars.append(sc)
    sp = torch.from_numpy(np.stack(spatials, axis=0))
    sc = torch.from_numpy(np.stack(scalars, axis=0))
    return sp, sc


class Trainer:
    def __init__(self, cfg: TrainConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.device = resolve_device(cfg.device)
        obs_spec = ObsSpec(fov=cfg.fov, n_pred=cfg.n_pred)
        self.net = PolicyNet(
            n_spatial_channels=obs_spec.num_spatial_channels,
            n_scalars=obs_spec.num_scalar_features,
            n_actions=5,
            fov=cfg.fov,
        ).to(self.device)
        try:
            self.optim = torch.optim.NAdam(self.net.parameters(), lr=cfg.lr)
        except AttributeError:
            self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.obs_spec = obs_spec
        self.episode = 0
        self.metrics_log_path = os.path.join(cfg.log_dir, f"train_metrics_{int(time.time())}.csv")
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        # Write CSV header.
        with open(self.metrics_log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "episode", "kind", "loss", "value_loss", "actor_loss", "valid_loss",
                "bc_loss", "entropy", "return_mean", "advantage_mean",
                "valid_rate", "goals_reached", "n_agents", "size", "density",
                "corridor_len", "wall_ms",
            ])
        self.log_path = os.path.join(cfg.log_dir, "train.log")
        self._log(f"Trainer initialized on device={self.device}")

    def _log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def _current_lr(self) -> float:
        # Paper V.B.1: "decay learning rate proportionally to the inverse square
        # root of the episode count".
        ref = max(1, self.cfg.lr_decay_ref_episode)
        return self.cfg.lr / math.sqrt(max(1, self.episode / ref))

    def _apply_lr(self) -> None:
        lr = self._current_lr()
        for pg in self.optim.param_groups:
            pg["lr"] = lr

    def _rl_episode(self) -> dict[str, float]:
        cfg = self.cfg
        grid, task, corridors, cell_to_id = _random_env(cfg, self.rng)
        n = grid.n_agents
        builder = ObservationBuilder(grid, task.goals, corridors, cell_to_id, self.obs_spec)

        # Per-agent trajectories.
        traj_logits: list[list[torch.Tensor]] = [[] for _ in range(n)]
        traj_values: list[list[torch.Tensor]] = [[] for _ in range(n)]
        traj_actions: list[list[int]] = [[] for _ in range(n)]
        traj_rewards: list[list[float]] = [[] for _ in range(n)]
        traj_valid: list[list[np.ndarray]] = [[] for _ in range(n)]

        hidden = self.net.init_hidden(n, self.device)
        valid_choose_total, valid_choose_correct = 0, 0
        T = cfg.rl_episode_length
        for t in range(T):
            sp, sc = _batch_obs(builder, n)
            sp = sp.to(self.device)
            sc = sc.to(self.device)
            logits, values, hidden = self.net(sp, sc, hidden)
            valid = compute_valid_actions(grid, corridors, cell_to_id)
            valid_t = torch.from_numpy(valid).to(self.device)
            # Sample only from valid actions (paper: "actions are sampled from a
            # list of valid actions"). We mask logits and re-normalize.
            masked = logits.clone()
            masked[valid_t == 0] = -1e9
            probs = torch.softmax(masked, dim=-1)
            # Fallback if a row has all invalid (shouldn't happen — stay always valid),
            # sample stay.
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
            actions_np = np.asarray(actions, dtype=np.int64)

            for i in range(n):
                traj_logits[i].append(logits[i])
                traj_values[i].append(values[i])
                traj_actions[i].append(int(actions_np[i]))
                traj_valid[i].append(valid[i].copy())
                if valid[i, int(actions_np[i])] == 1.0:
                    valid_choose_correct += 1
                valid_choose_total += 1

            rewards, arrived, done = task.step(actions_np)
            for i in range(n):
                traj_rewards[i].append(float(rewards[i]))
            if done:
                break

        # Bootstrap: value at final obs (0 if terminated).
        # Get one more forward pass on final state (before optimizer step so we can
        # detach). We approximate bootstrap by V of last-state.
        sp, sc = _batch_obs(builder, n)
        sp = sp.to(self.device)
        sc = sc.to(self.device)
        with torch.no_grad():
            _, final_values, _ = self.net(sp, sc, hidden)

        # Compute loss per agent, sum.
        total_loss = torch.zeros((), device=self.device)
        metrics: dict[str, float] = {"value_loss": 0.0, "actor_loss": 0.0, "valid_loss": 0.0,
                                     "entropy": 0.0, "return_mean": 0.0, "advantage_mean": 0.0}
        for i in range(n):
            logits_i = torch.stack(traj_logits[i], dim=0)
            values_i = torch.stack(traj_values[i], dim=0)
            actions_i = torch.tensor(traj_actions[i], device=self.device, dtype=torch.long)
            rewards_i = torch.tensor(traj_rewards[i], device=self.device, dtype=torch.float32)
            valid_i = torch.from_numpy(np.stack(traj_valid[i], axis=0)).to(self.device)
            loss_i, m_i = a3c_loss(
                logits_i, values_i, actions_i, rewards_i, valid_i,
                final_values[i].detach(),
                gamma=cfg.gamma,
                entropy_weight=cfg.entropy_weight,
                value_weight=cfg.value_weight,
                actor_weight=cfg.actor_weight,
                valid_weight=cfg.valid_weight,
            )
            total_loss = total_loss + loss_i
            for k in metrics:
                metrics[k] += m_i[k]
        for k in metrics:
            metrics[k] /= n

        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optim.step()

        metrics["loss"] = float(total_loss.item())
        metrics["valid_rate"] = (valid_choose_correct / valid_choose_total) if valid_choose_total else 0.0
        metrics["goals_reached"] = int(task.goals_reached_count.sum())
        metrics["n_agents"] = n
        return metrics

    def _il_episode(self) -> dict[str, float]:
        cfg = self.cfg
        grid, task, corridors, cell_to_id = _random_env(cfg, self.rng)
        n = grid.n_agents
        # Get expert plans.
        starts = [tuple(grid.positions[i]) for i in range(n)]
        goals = [tuple(task.goals[i]) for i in range(n)]
        expert_actions = expert_plan(
            grid.obstacle_map, starts, goals, horizon=cfg.il_episode_length, rng=self.rng
        )
        builder = ObservationBuilder(grid, task.goals, corridors, cell_to_id, self.obs_spec)
        hidden = self.net.init_hidden(n, self.device)
        traj_logits: list[list[torch.Tensor]] = [[] for _ in range(n)]
        traj_expert: list[list[int]] = [[] for _ in range(n)]
        traj_valid: list[list[np.ndarray]] = [[] for _ in range(n)]
        T = cfg.il_episode_length
        for t in range(T):
            sp, sc = _batch_obs(builder, n)
            sp = sp.to(self.device)
            sc = sc.to(self.device)
            logits, values, hidden = self.net(sp, sc, hidden)
            valid = compute_valid_actions(grid, corridors, cell_to_id)
            actions_np = np.array([expert_actions[i][t] for i in range(n)], dtype=np.int64)
            for i in range(n):
                traj_logits[i].append(logits[i])
                traj_expert[i].append(int(actions_np[i]))
                traj_valid[i].append(valid[i].copy())
            rewards, arrived, done = task.step(actions_np)
            if done:
                break
        total_loss = torch.zeros((), device=self.device)
        metrics = {"bc_loss": 0.0, "valid_loss": 0.0}
        for i in range(n):
            logits_i = torch.stack(traj_logits[i], dim=0)
            expert_i = torch.tensor(traj_expert[i], device=self.device, dtype=torch.long)
            valid_i = torch.from_numpy(np.stack(traj_valid[i], axis=0)).to(self.device)
            loss_i, m_i = bc_loss(logits_i, expert_i, valid_i, valid_weight=cfg.valid_weight)
            total_loss = total_loss + loss_i
            for k in metrics:
                metrics[k] += m_i[k]
        for k in metrics:
            metrics[k] /= n
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.optim.step()
        metrics["loss"] = float(total_loss.item())
        metrics["goals_reached"] = int(task.goals_reached_count.sum())
        metrics["n_agents"] = n
        return metrics

    def _record(self, kind: str, metrics: dict[str, Any], env_desc: dict, wall_ms: float) -> None:
        with open(self.metrics_log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self.episode, kind,
                metrics.get("loss", 0.0),
                metrics.get("value_loss", 0.0),
                metrics.get("actor_loss", 0.0),
                metrics.get("valid_loss", 0.0),
                metrics.get("bc_loss", 0.0),
                metrics.get("entropy", 0.0),
                metrics.get("return_mean", 0.0),
                metrics.get("advantage_mean", 0.0),
                metrics.get("valid_rate", 0.0),
                metrics.get("goals_reached", 0),
                metrics.get("n_agents", 0),
                env_desc.get("size", 0),
                env_desc.get("density", 0.0),
                env_desc.get("corridor_len", 0),
                wall_ms,
            ])

    def save_checkpoint(self, tag: str = "latest") -> str:
        path = os.path.join(self.cfg.ckpt_dir, f"primal2_{tag}.pt")
        torch.save({
            "episode": self.episode,
            "model": self.net.state_dict(),
            "optim": self.optim.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(s["model"])
        try:
            self.optim.load_state_dict(s["optim"])
        except Exception:
            pass
        self.episode = int(s.get("episode", 0))

    def run(self, deadline_ts: float | None = None) -> None:
        cfg = self.cfg
        start_ts = time.time()
        while self.episode < cfg.total_episodes:
            if deadline_ts is not None and time.time() >= deadline_ts:
                self._log(f"[ep {self.episode}] deadline reached, stopping.")
                break
            self.episode += 1
            self._apply_lr()
            wall_t0 = time.time()
            do_il = self.rng.random() < cfg.il_prob
            if do_il:
                metrics = self._il_episode()
                kind = "IL"
            else:
                metrics = self._rl_episode()
                kind = "RL"
            wall_ms = (time.time() - wall_t0) * 1000
            env_desc: dict = {}  # We don't have easy access to the env desc after run;
                                 # metrics carry n_agents and we can add later if needed.
            self._record(kind, metrics, env_desc, wall_ms)
            if self.episode % cfg.log_every_episodes == 0 or self.episode < 10:
                elapsed = time.time() - start_ts
                lr = self._current_lr()
                self._log(
                    f"[ep {self.episode:5d}] kind={kind} loss={metrics.get('loss', 0):.3f} "
                    f"vr={metrics.get('valid_rate', float('nan')):.3f} "
                    f"goals={metrics.get('goals_reached', 0)} n={metrics.get('n_agents', 0)} "
                    f"lr={lr:.2e} elapsed={elapsed:.0f}s ms/ep={wall_ms:.0f}"
                )
            if self.episode % cfg.ckpt_every_episodes == 0:
                p = self.save_checkpoint(tag="latest")
                self._log(f"[ep {self.episode}] checkpoint -> {p}")
        # Final save.
        p = self.save_checkpoint(tag=f"final_ep{self.episode}")
        p2 = self.save_checkpoint(tag="latest")
        self._log(f"training done. checkpoints saved to {p} and {p2}")
