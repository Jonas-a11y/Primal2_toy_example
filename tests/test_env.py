"""Quick sanity smoke: generate a maze, place agents, step randomly."""
import numpy as np
from primal2_toy.env.maze import generate_maze
from primal2_toy.env.grid import GridWorld
from primal2_toy.env.lmapf import LMAPFTask, Config
from primal2_toy.env.corridor import analyze


def test_env_smoke():
    rng = np.random.default_rng(0)
    m = generate_maze(15, 0.3, 5, rng)
    assert m.shape == (15, 15)
    assert m.dtype == np.uint8
    empties = np.argwhere(m == 0)
    assert len(empties) >= 8

    starts = [tuple(empties[i]) for i in rng.choice(len(empties), size=8, replace=False)]
    grid = GridWorld(m, starts)
    cfg = Config(size=15, density=0.3, typical_corridor_length=5, n_agents=8, episode_length=32)
    task = LMAPFTask(grid, cfg, rng)

    for _ in range(32):
        actions = rng.integers(0, 5, size=8)
        rewards, arrived, done = task.step(actions)
        assert rewards.shape == (8,)
        if done:
            break
    print("env_smoke OK")


def test_corridor_smoke():
    # Hand-crafted map: horizontal corridor.
    #   . . . # # # . . .
    #   . # . . . . . # .
    #   . . . # # # . . .
    m = np.zeros((3, 9), dtype=np.uint8)
    m[0, 3:6] = 1
    m[1, 1] = 1
    m[1, 7] = 1
    m[2, 3:6] = 1
    corridors, cell_to_id = analyze(m)
    assert len(corridors) > 0
    print(f"corridor_smoke OK: {len(corridors)} corridors")


if __name__ == "__main__":
    test_env_smoke()
    test_corridor_smoke()
