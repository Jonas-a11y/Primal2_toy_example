# PRIMAL2 Toy Example

A small, from-scratch reimplementation of **PRIMAL2** (Damani, Luo, Wenzel, Sartoretti — *"PRIMAL2: Pathfinding via Reinforcement and Imitation Multi-Agent Learning - Lifelong"*, RA-L 2021, [arXiv:2010.08184](https://arxiv.org/abs/2010.08184)) intended for a seminar demo.

The paper trains a fully decentralized, communication-free policy for **lifelong multi-agent path finding (LMAPF)** in dense, corridor-heavy grid worlds. Each agent observes only an 11×11 local FOV enriched with A* path hints and corridor-structure channels, and is trained with a mix of A3C, imitation learning from a centralized expert, and a supervised "convention" loss.

This repo re-implements the paper *as faithfully as is feasible* at a small training scale (a few hours on a laptop), so you can run it live during a seminar.

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

- `docs/superpowers/specs/2026-07-02-primal2-toy-example-design.md`
- `logs/dev.md`

## Faithfulness to the paper

- **Network:** exact — Section IV.D.
- **Observation channels:** all 13 as described in Section IV.A + Fig. 3.
- **Losses:** value + actor with entropy bonus (Eq. 1), advantage from bootstrapped value (Eq. 2), valid/BCE loss (Eq. 3), behavior cloning (Eq. 4).
- **Reward structure:** `-0.3` off goal, `+5` on goal, `-2` on collision (Section IV.C).
- **Training recipe:** NAdam, lr `2e-5`, inverse-sqrt decay, γ=0.95, RL episodes 256 steps, IL episodes 64 steps, RL/IL ratio 0.5 (Section V.B).
- **Env randomization:** size, obstacle density, corridor length per episode.

## Documented deviations

- Distributed backend: `torch.multiprocessing` instead of Ray, currently **1 worker** (paper uses 9). This is the main gap.
- Expert planner: prioritized-planning multi-agent A* stand-in for ODrM* (same role, simpler code, still produces valid demonstrations).
- Scale: world 10–20 (paper 20–160), 6–8 agents/env (paper up to 2048).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy torch pygame matplotlib tqdm

# Train (6h on M2 Pro / MPS, ~22k episodes):
PYTHONPATH=. python -m primal2_toy.train.main \
    --episodes 50000 --deadline-hours 6 --device mps \
    --n-agents 6 --seed 42 --log-every 25 --ckpt-every 500

# Watch training progress:
PYTHONPATH=. python -m primal2_toy.eval.monitor logs/train_metrics_*.csv

# Periodically evaluate the latest checkpoint:
PYTHONPATH=. python -m primal2_toy.eval.watchdog --agents 6 --seeds 7 42 123 555

# Live demo:
python demo.py --checkpoint checkpoints/primal2_latest.pt --agents 6 --seed 42 --fps 4

# Baselines for comparison:
PYTHONPATH=. python -m primal2_toy.eval.baselines --baseline random --agents 6
PYTHONPATH=. python -m primal2_toy.eval.baselines --baseline greedy_astar --agents 6
```

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
├── primal2.pdf                          # source paper (already here)
├── README.md
├── docs/superpowers/specs/              # design docs
├── logs/                                # dev log, training CSVs, screenshots
├── checkpoints/                         # saved model states
├── primal2_toy/
│   ├── env/         grid.py maze.py corridor.py lmapf.py
│   ├── obs/         builder.py astar.py corridor_maps.py
│   ├── expert/      prioritized_astar.py
│   ├── policy/      network.py losses.py
│   ├── train/       config.py trainer.py validity.py main.py
│   └── eval/        rollout.py visualizer.py headless.py baselines.py monitor.py watchdog.py render_frame.py
├── demo.py
└── tests/                               # smoke tests for each module
```
