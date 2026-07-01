"""Render a single frame of the demo to a PNG for inspection."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
import torch
import argparse

from primal2_toy.policy.network import PolicyNet
from primal2_toy.obs.builder import ObsSpec
from primal2_toy.eval.rollout import build_scenario, Rollout
from primal2_toy.eval.visualizer import draw, draw_obs_panel, HUD_BG, HUD_FG, BG, PANEL_BG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--out", type=str, default="logs/demo_frame.png")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    pygame.init()
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features, n_actions=5, fov=11,
    )
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    net.load_state_dict(state["model"])
    net.eval()
    grid, task, corridors, cell_to_id = build_scenario(
        args.size, args.density, args.corridor_length, args.agents, args.seed
    )
    dev = torch.device(args.device)
    rollout = Rollout(net, grid, task, corridors, cell_to_id, dev, obs_spec)
    for _ in range(args.steps):
        rollout.step()

    cell = 28
    w_grid = grid.w * cell
    h_grid = grid.h * cell
    hud_h = 80
    panel_w = 3 * (11 * 8 + 6) + 20
    surf = pygame.Surface((w_grid + panel_w, max(h_grid + hud_h, 4 * (11 * 8 + 22) + 40)))
    surf.fill(BG)
    draw(surf, grid.obstacle_map, grid.positions, task.goals, cell)
    pygame.draw.rect(surf, HUD_BG, (0, h_grid, w_grid, hud_h))
    font = pygame.font.SysFont("Menlo", 16)
    font_small = pygame.font.SysFont("Menlo", 11)
    throughput = rollout.total_arrivals / max(1, rollout.step_idx)
    lines = [
        f"step {rollout.step_idx:4d}  throughput {throughput:.3f}  arrivals {rollout.total_arrivals}",
        f"world {grid.h}x{grid.w}  agents {grid.n_agents}  seed {args.seed}",
        f"[SPACE] pause  [R] reset  [V] obs panel  [click] select agent",
    ]
    for i, s in enumerate(lines):
        img = font.render(s, True, HUD_FG)
        surf.blit(img, (10, h_grid + 8 + i * 22))
    pygame.draw.rect(surf, PANEL_BG, (w_grid, 0, panel_w, surf.get_height()))
    rollout.builder.refresh_step_cache()
    spatial, scalars = rollout.builder.build(0)
    surf.blit(font.render(f"agent 0 obs channels", True, (30, 30, 30)), (w_grid + 10, 8))
    draw_obs_panel(surf, spatial, (w_grid + 10, 32), font_small, cell=8)
    pygame.image.save(surf, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
