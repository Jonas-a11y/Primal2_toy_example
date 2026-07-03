# PRIMAL2 Toy Example

A small, from-scratch reimplementation of **PRIMAL2** (Damani, Luo, Wenzel, Sartoretti — *"PRIMAL2: Pathfinding via Reinforcement and Imitation Multi-Agent Learning - Lifelong"*, RA-L 2021, [arXiv:2010.08184](https://arxiv.org/abs/2010.08184)) intended for a seminar demo.

The paper trains a fully decentralized, communication-free policy for **lifelong multi-agent path finding (LMAPF)** in dense, corridor-heavy grid worlds. Each agent observes only an 11×11 local FOV enriched with A* path hints and corridor-structure channels, and is trained with a mix of A3C, imitation learning from a centralized expert, and a supervised "convention" loss.

This repo re-implements the paper *as faithfully as is feasible* at a small training scale (a few hours on a laptop), so you can run it live during a seminar.

![comparison bar chart](docs/images/comparison.png)

## Headline result

On a held-out benchmark of 20 seeds × 256 steps, 15×15 world, 30% density, corridor length 5, 6 agents:

| Method | Throughput (arrivals / step) | vs greedy A* |
| --- | ---:| ---:|
| random | 0.006 | 0.3× |
| greedy A* (independent) | 0.019 | 1.0× |
| **PRIMAL2 (learned, greedy)** | **0.053** | **2.8×** |
| **PRIMAL2 (learned, sampled)** | **0.140** | **7.5×** |

The trained policy also **never deadlocks** — greedy A*'s worst seed gets 0 arrivals; PRIMAL2's worst gets 0.059.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the full write-up.

## What's in this repo

| Module | What it does |
| --- | --- |
| `primal2_toy/env/` | 2D 4-connected grid, wall-drop maze generator, corridor decomposition, LMAPF task manager. |
| `primal2_toy/obs/` | Builds the 13-channel FOV observation (obstacles, own goal, other agents, other goals, A* path-length map, ΔX/ΔY/blocking, three future-position prediction maps) + 3 goal-vector scalars. |
| `primal2_toy/expert/` | Prioritized-planning space-time A* to generate collision-free expert demonstrations for IL episodes. |
| `primal2_toy/policy/` | Paper's exact network (2× VGG blocks → 1×1 conv → concat with goal-FC → 2× FC → LSTM with residual → π, V) plus the four losses (value, actor+entropy, valid/BCE, BC). |
| `primal2_toy/train/` | Training loop: env randomization per episode, 50/50 RL/IL dispatch, NAdam with inverse-sqrt LR decay, checkpointing, CSV logging. |
| `primal2_toy/eval/` | Pygame visualizer, headless evaluator, random & greedy-A* baselines, live watchdog. |
| `demo.py` | Live-demo entry point. |

Design notes and dev log:

- [`docs/superpowers/specs/2026-07-02-primal2-toy-example-design.md`](docs/superpowers/specs/2026-07-02-primal2-toy-example-design.md)
- [`docs/dev-log.md`](docs/dev-log.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)

## Faithfulness to the paper

- **Network:** exact — Section IV.D.
- **Observation channels:** all 13 as described in Section IV.A + Fig. 3.
- **Losses:** value + actor with entropy bonus (Eq. 1), advantage from bootstrapped value (Eq. 2), valid/BCE loss (Eq. 3), behavior cloning (Eq. 4).
- **Reward structure:** `-0.3` off goal, `+5` on goal, `-2` on collision (Section IV.C).
- **Training recipe:** NAdam, lr `5e-5` (paper uses `2e-5`; we bumped for the single-worker setting), inverse-sqrt decay, γ=0.95, RL episodes 256 steps, IL episodes 64 steps, RL/IL ratio 0.5 (Section V.B).
- **Env randomization:** size, obstacle density, corridor length per episode.

## Documented deviations

- Distributed backend: single Python process (paper uses Ray with 9 workers). This is the main gap.
- Expert planner: prioritized-planning multi-agent A* stand-in for ODrM* (same role, simpler code, still produces valid demonstrations).
- Scale: world 10–20 (paper 20–160), 6–8 agents/env (paper up to 2048).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy torch pygame matplotlib tqdm

# Live demo with the shipped trained checkpoint:
python demo.py --checkpoint checkpoints/primal2_final.pt --agents 6 --seed 42 --fps 4

# Empty-room variant (no obstacles, no corridors):
python demo.py --checkpoint checkpoints/primal2_final.pt --agents 6 --seed 42 --no-corridors --fps 4

# Baselines for comparison:
PYTHONPATH=. python -m primal2_toy.eval.baselines --baseline random --agents 6
PYTHONPATH=. python -m primal2_toy.eval.baselines --baseline greedy_astar --agents 6

# Rerun the held-out comparison:
PYTHONPATH=. python -m primal2_toy.eval.compare \
    --checkpoint checkpoints/primal2_final.pt \
    --agents 6 --steps 256 --device cpu \
    --seeds 7 42 123 555 2024 8 91 314 777 1000 \
            33 66 200 400 800 1234 5678 9101 2222 3333

# Retrain from scratch (takes several hours on M2 Pro / MPS or a modern GPU):
PYTHONPATH=. python -m primal2_toy.train.main \
    --episodes 50000 --deadline-hours 5 --device mps \
    --n-agents 6 --seed 44 --log-every 25 --ckpt-every 400
```

## Reproducing the paper's PRIMAL2 numbers

The shipped checkpoint was trained on small worlds only (10-20, 6 agents). To
recreate the paper's Fig. 4 / Fig. 5 (PRIMAL2 line only) you need to (a) train
on the paper's wider env randomization and (b) sweep the same axes at eval time.

### 1. Retrain with the paper's env-randomization ranges

Paper Section V.B.2: sizes 10-70, obstacle density 0.2-0.7, typical corridor
length 3-21. Enable with `--paper-ranges`:

```bash
PYTHONPATH=. python -m primal2_toy.train.main \
    --paper-ranges --n-agents 8 --device cuda \
    --episodes 35000 --seed 42 --log-every 50 --ckpt-every 1000
```

Wall-clock is ~10 h on a modern GPU. Consider `--num-workers` (WIP) or
warm-starting from the shipped checkpoint (`--warmstart-weights checkpoints/primal2_final.pt`)
if you want a faster path.

### 2. Sweep the paper's evaluation grid

Fig. 5 (LMAPF throughput vs. team size):

```bash
PYTHONPATH=. python -m primal2_toy.eval.sweep \
    --checkpoint checkpoints/primal2_paper_ranges.pt \
    --out logs/sweep_lmapf.csv --mode lmapf \
    --sizes 20 40 80 --densities 0.3 --corridors 10 \
    --team-sizes 4 8 16 32 64 128 --n-seeds 10 --device cuda
PYTHONPATH=. python -m primal2_toy.eval.sweep_plots \
    --sweep-csv logs/sweep_lmapf.csv --out docs/images/sweep
```

Fig. 4 (one-shot success rate + path length):

```bash
PYTHONPATH=. python -m primal2_toy.eval.sweep \
    --checkpoint checkpoints/primal2_paper_ranges.pt \
    --out logs/sweep_oneshot.csv --mode oneshot \
    --sizes 20 40 --densities 0.3 --corridors 10 \
    --team-sizes 4 8 16 32 64 128 --n-seeds 10 --device cuda
PYTHONPATH=. python -m primal2_toy.eval.sweep_plots \
    --sweep-csv logs/sweep_oneshot.csv --out docs/images/sweep_oneshot
```

The paper uses 50 seeds per config. Scale `--n-seeds` to your compute budget.
Timestep budgets default to the paper's per-size values (Section VI.A/B). Add
`--greedy` to use argmax actions; sampled (default) is usually stronger.

### Demo controls

- `SPACE` — pause/resume
- `V` — toggle side panel showing the 11 spatial obs channels for the selected agent
- `Click on an agent` — select agent for the obs panel
- `R` — reset with a new random map
- `+ / -` — adjust FPS
- `ESC` — quit

## Repository layout

```
primal2_toy_example/
├── primal2.pdf                          # source paper
├── README.md
├── docs/
│   ├── EVALUATION.md                    # results write-up
│   ├── dev-log.md                       # narrative of the 5 training runs
│   ├── images/                          # comparison chart, training curves, screenshot
│   └── superpowers/specs/               # design doc
├── checkpoints/
│   └── primal2_final.pt                 # shipped trained model
├── primal2_toy/
│   ├── env/         grid.py maze.py corridor.py lmapf.py
│   ├── obs/         builder.py astar.py corridor_maps.py
│   ├── expert/      prioritized_astar.py
│   ├── policy/      network.py losses.py
│   ├── train/       config.py trainer.py validity.py main.py
│   └── eval/        rollout.py visualizer.py headless.py baselines.py monitor.py watchdog.py render_frame.py compare.py comparison_plot.py plots.py
├── demo.py
└── tests/                               # smoke tests for each module
```

Training-time artifacts (`logs/`, intermediate checkpoints) are gitignored — regenerable by running the training loop.
