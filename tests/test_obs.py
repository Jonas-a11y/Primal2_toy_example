"""Smoke test for observation builder."""
import numpy as np

from primal2_toy.env.grid import GridWorld
from primal2_toy.env.maze import generate_maze
from primal2_toy.env.corridor import analyze
from primal2_toy.env.lmapf import LMAPFTask, Config
from primal2_toy.obs.builder import ObservationBuilder, ObsSpec


def main():
    rng = np.random.default_rng(1)
    m = generate_maze(15, 0.3, 5, rng)
    empties = np.argwhere(m == 0)
    starts = [tuple(empties[i]) for i in rng.choice(len(empties), size=6, replace=False)]
    grid = GridWorld(m, starts)
    cfg = Config(size=15, n_agents=6, episode_length=8)
    task = LMAPFTask(grid, cfg, rng)
    corridors, cell_to_id = analyze(m)
    builder = ObservationBuilder(grid, task.goals, corridors, cell_to_id, ObsSpec(fov=11, n_pred=3))
    for i in range(6):
        spatial, scalars = builder.build(i)
        assert spatial.shape == (11, 11, 11), spatial.shape
        assert scalars.shape == (3,)
        assert np.isfinite(spatial).all() and np.isfinite(scalars).all()
    print(f"obs_smoke OK — spatial {spatial.shape}, scalars {scalars.shape}")


if __name__ == "__main__":
    main()
