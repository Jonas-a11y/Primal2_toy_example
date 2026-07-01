# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=64, seeds=[7, 42]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 400)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0000 | 0.0000 – 0.0000 | 0 | 128 |
| greedy_astar | 0.0391 | 0.0312 – 0.0469 | 5 | 128 |
| random | 0.0078 | 0.0000 – 0.0156 | 1 | 128 |
