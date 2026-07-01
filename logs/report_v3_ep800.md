# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=256, seeds=[7, 42, 123, 555]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 800)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0215 | 0.0078 – 0.0391 | 22 | 1024 |
| greedy_astar | 0.0107 | 0.0078 – 0.0156 | 11 | 1024 |
| random | 0.0029 | 0.0000 – 0.0078 | 3 | 1024 |
