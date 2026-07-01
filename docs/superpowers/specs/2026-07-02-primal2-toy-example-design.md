# PRIMAL2 Toy Example — Design

**Date:** 2026-07-02
**Purpose:** Live-demo-ready reimplementation of PRIMAL2 (Damani et al., RA-L 2021) at small scale, faithful to the paper.

## Scope

Full-fidelity to the paper except training scale. Included:

- **Network:** paper's exact architecture (2 VGG blocks → 1×1 conv → concat with goal-FC → 2× FC → LSTM w/ residual → π, V heads).
- **Observation:** all 13 channels (11×11 FOV) — obstacles, own goal, other agents, other goals, single-agent A* path-length, ΔX, ΔY, blocking, 3 future-position maps — plus 3 goal-vector scalars.
- **Losses:** value + actor + entropy + valid (convention) + BC on IL episodes.
- **Training:** A3C-style parallel workers, 50/50 RL/IL episode dispatch, NAdam, lr 2e-5 with inverse-sqrt decay, γ=0.95.
- **Expert for IL:** prioritized-planning multi-agent A* (stand-in for ODrM*; produces valid collision-free plans, same *role* in the pipeline).
- **Env randomization:** world size, obstacle density, corridor length sampled per episode.

Scaled down: world 10–20, 8 agents/worker, 4 workers, ~30k episodes total. Everything else = paper.

## Architecture

```
primal2_toy/
├── env/          grid.py maze.py corridor.py lmapf.py
├── obs/          builder.py astar.py corridor_maps.py
├── expert/       prioritized_astar.py
├── policy/       network.py losses.py
├── train/        config.py worker.py main.py
└── eval/         rollout.py visualizer.py
demo.py
tests/
```

## Data flow (inference)

`env.state ─► obs.build(agent_id) ─► policy.act(obs, hidden) ─► env.step({agent:action})`

## Reward (Section IV.C exact)

`-0.3` off-goal, `+5` on reaching goal, `-2` on collision, invalid actions masked at sampling.

## Losses (Section IV.D / V.A exact)

- `L_value = mean((R_t - V(o_t))^2)` with R_t discounted at γ=0.95.
- `L_actor = mean(σ_H·H(π) - log π(a_t)·A(o_t,a_t))`, A = r + γV(o_{t+1}) - V(o_t).
- `L_valid = -mean(Σ_i BCE(sigmoid(π_logit_i), v_i(t)))` — per-action Bernoulli validity target.
- `L_bc = -mean(log π(a_t^expert))` on IL episodes.
- `L_final_RL = α·L_value + β·L_actor + ζ·L_valid`, `L_final_IL = L_bc + ζ·L_valid`.
- Weights: α=0.5, β=1.0, ζ=0.5, σ_H=0.01.

## Validity flags per timestep

- Move-into-obstacle → invalid
- Move-into-another-agent → invalid
- Immediate reversal → invalid
- Corridor entry against blocking → invalid
- Reverse-inside-corridor → invalid

## Deliverables (this 8-hour push)

1. Working env + obs + expert + net + losses + training loop, all with sanity tests.
2. A checkpointed model trained as long as budget allows.
3. `demo.py` with Pygame visualizer that loads the checkpoint and renders a seeded scenario.
4. Short evaluation report (metrics + notes on faithfulness and limitations).

## Deviations from paper (all documented)

- Ray → `torch.multiprocessing`.
- ODrM* expert → prioritized-planning multi-agent A*.
- Workers 9 → 4; world sizes 10–70 → 10–20; episodes 35k → whatever fits budget.
