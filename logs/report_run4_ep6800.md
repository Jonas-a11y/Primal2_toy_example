# PRIMAL2 Toy — Evaluation Report

**Scenario:** size=15, density=0.3, corridor_length=5, agents=6, steps/scenario=256, seeds=[7, 42, 123, 555, 2024, 8, 91, 314, 777, 1000]
**Checkpoint:** `checkpoints/primal2_latest.pt` (episode 6800)

| Method | Throughput (mean) | Throughput (min–max) | Total arrivals | Total steps |
|---|---|---|---|---|
| PRIMAL2 (learned) | 0.0457 | 0.0078 – 0.0898 | 117 | 2560 |
| PRIMAL2 (sampled) | 0.1008 | 0.0430 – 0.2031 | 258 | 2560 |
| greedy_astar | 0.0121 | 0.0000 – 0.0312 | 31 | 2560 |
| random | 0.0051 | 0.0000 – 0.0195 | 13 | 2560 |
