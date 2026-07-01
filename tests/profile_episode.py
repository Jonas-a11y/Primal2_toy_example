"""Profile a single RL episode to find the bottleneck."""
import time
import numpy as np
import torch
from primal2_toy.train.config import TrainConfig
from primal2_toy.train.trainer import Trainer, _random_env, _batch_obs
from primal2_toy.obs.builder import ObservationBuilder, ObsSpec
from primal2_toy.train.validity import compute_valid_actions


def main():
    cfg = TrainConfig(total_episodes=1, device="cpu", n_agents=8)
    tr = Trainer(cfg, seed=0)
    grid, task, corridors, cell_to_id = _random_env(cfg, tr.rng)
    n = grid.n_agents
    builder = ObservationBuilder(grid, task.goals, corridors, cell_to_id, tr.obs_spec)
    hidden = tr.net.init_hidden(n, tr.device)

    t_obs, t_valid, t_fwd, t_step = 0.0, 0.0, 0.0, 0.0
    T = 128
    for t in range(T):
        t0 = time.time(); sp, sc = _batch_obs(builder, n); t_obs += time.time() - t0
        t0 = time.time(); valid = compute_valid_actions(grid, corridors, cell_to_id); t_valid += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            logits, values, hidden = tr.net(sp, sc, hidden)
        t_fwd += time.time() - t0
        actions = np.random.randint(0, 5, n)
        t0 = time.time(); task.step(actions); t_step += time.time() - t0
    print(f"T={T}, n={n}")
    print(f"obs: {t_obs*1000:.0f} ms  ({t_obs/T*1000:.2f} ms/step)")
    print(f"valid: {t_valid*1000:.0f} ms  ({t_valid/T*1000:.2f} ms/step)")
    print(f"forward: {t_fwd*1000:.0f} ms ({t_fwd/T*1000:.2f} ms/step)")
    print(f"env.step: {t_step*1000:.0f} ms ({t_step/T*1000:.2f} ms/step)")

if __name__ == "__main__":
    main()
