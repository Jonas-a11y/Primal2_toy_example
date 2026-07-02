"""Pygame visualizer for a seeded PRIMAL2 toy scenario.

Displays the gridworld: obstacles grey, agents colored circles, goals matching
hollow squares. Shows a small HUD with step count, throughput (arrivals/step),
and the current world parameters.

Optional side panel (toggle with V) shows the 13 observation channels for the
currently-selected agent (click on an agent to select).
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pygame
import torch

from ..policy.network import PolicyNet
from ..obs.builder import ObsSpec
from .rollout import Rollout, build_scenario


# Colors.
BG = (245, 245, 245)
OBSTACLE = (60, 60, 60)
GRID_LINE = (210, 210, 210)
HUD_BG = (30, 30, 30)
HUD_FG = (240, 240, 240)
PANEL_BG = (250, 250, 250)
CHANNEL_LABELS = [
    "obstacles", "own goal", "others", "others' goals", "path length",
    "ΔX", "ΔY", "blocking", "future t+1", "future t+2", "future t+3",
]


def agent_color(i: int) -> tuple[int, int, int]:
    palette = [
        (220, 40, 40), (40, 40, 220), (40, 180, 40), (240, 180, 20),
        (180, 40, 200), (20, 180, 200), (240, 100, 20), (100, 200, 40),
        (200, 200, 40), (100, 100, 240), (240, 40, 140), (40, 200, 140),
    ]
    return palette[i % len(palette)]


def load_net(ckpt_path: str, device: torch.device, obs_spec: ObsSpec) -> PolicyNet:
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features,
        n_actions=5,
        fov=obs_spec.fov,
    ).to(device)
    s = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(s["model"])
    net.eval()
    return net


def draw(
    surface: pygame.Surface,
    obstacle_map: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    cell_size: int,
    origin: tuple[int, int] = (0, 0),
) -> None:
    h, w = obstacle_map.shape
    ox, oy = origin
    # background
    pygame.draw.rect(surface, BG, (ox, oy, w * cell_size, h * cell_size))
    # obstacles
    for r in range(h):
        for c in range(w):
            if obstacle_map[r, c]:
                pygame.draw.rect(surface, OBSTACLE, (ox + c * cell_size, oy + r * cell_size, cell_size, cell_size))
    # grid
    for r in range(h + 1):
        pygame.draw.line(surface, GRID_LINE, (ox, oy + r * cell_size), (ox + w * cell_size, oy + r * cell_size))
    for c in range(w + 1):
        pygame.draw.line(surface, GRID_LINE, (ox + c * cell_size, oy), (ox + c * cell_size, oy + h * cell_size))
    # goals (hollow squares)
    n = positions.shape[0]
    for i in range(n):
        gr, gc = int(goals[i, 0]), int(goals[i, 1])
        color = agent_color(i)
        pygame.draw.rect(
            surface, color,
            (ox + gc * cell_size + 3, oy + gr * cell_size + 3, cell_size - 6, cell_size - 6),
            width=3,
        )
    # agents (filled circles)
    for i in range(n):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        color = agent_color(i)
        cx = ox + c * cell_size + cell_size // 2
        cy = oy + r * cell_size + cell_size // 2
        pygame.draw.circle(surface, color, (cx, cy), cell_size // 2 - 3)


def draw_obs_panel(
    surface: pygame.Surface,
    spatial: np.ndarray,  # (C, F, F)
    origin: tuple[int, int],
    font: pygame.font.Font,
    cell: int = 8,
) -> None:
    C, F, _ = spatial.shape
    ox, oy = origin
    per_row = 3
    padding = 6
    label_h = 16
    tile_w = F * cell + padding
    tile_h = F * cell + label_h + padding
    for c in range(C):
        rr = c // per_row
        cc = c % per_row
        tx = ox + cc * tile_w
        ty = oy + rr * tile_h
        # Label
        label = CHANNEL_LABELS[c] if c < len(CHANNEL_LABELS) else f"ch{c}"
        img = font.render(label, True, (30, 30, 30))
        surface.blit(img, (tx, ty))
        # Heatmap
        ch = spatial[c]
        mn = float(ch.min()); mx = float(ch.max())
        rng_val = mx - mn if mx > mn else 1.0
        for r in range(F):
            for cc2 in range(F):
                v = (ch[r, cc2] - mn) / rng_val if rng_val > 0 else 0.0
                v = float(np.clip(v, 0.0, 1.0))
                # Blue-to-red colormap.
                col = (int(255 * v), int(120 * (1 - v)), int(255 * (1 - v)))
                pygame.draw.rect(
                    surface, col,
                    (tx + cc2 * cell, ty + label_h + r * cell, cell, cell)
                )


def run_demo(args) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() and args.device in ("auto", "mps")
        else "cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda")
        else "cpu"
    )
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = load_net(args.checkpoint, device, obs_spec)
    density = 0.0 if args.no_corridors else args.density
    corridor_length = 1 if args.no_corridors else args.corridor_length
    grid, task, corridors, cell_to_id = build_scenario(
        size=args.size, density=density, corridor_length=corridor_length,
        n_agents=args.agents, seed=args.seed,
    )
    rollout = Rollout(net, grid, task, corridors, cell_to_id, device, obs_spec, greedy=args.greedy)

    pygame.init()
    pygame.display.set_caption("PRIMAL2 toy example")
    cell = 32
    w_grid = grid.w * cell
    h_grid = grid.h * cell
    hud_h = 80
    panel_w = 3 * (11 * 8 + 6) + 20  # room for 3 columns of 11×11 obs heatmaps
    panel_h = 4 * (11 * 8 + 16 + 6) + 30

    show_panel = False
    selected_agent = 0
    screen = pygame.display.set_mode((w_grid + panel_w, max(h_grid + hud_h, panel_h)))
    font = pygame.font.SysFont("Menlo, monospace", 18)
    font_small = pygame.font.SysFont("Menlo, monospace", 11)

    running = True
    paused = False
    fps = args.fps
    tick_ms_target = 1000 / max(1, fps)
    last_step = 0.0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Click to select agent.
                mx, my = ev.pos
                if mx < w_grid and my < h_grid:
                    r_click = my // cell
                    c_click = mx // cell
                    for i in range(grid.n_agents):
                        if int(grid.positions[i, 0]) == r_click and int(grid.positions[i, 1]) == c_click:
                            selected_agent = i
                            break
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_v:
                    show_panel = not show_panel
                elif ev.key == pygame.K_r:
                    grid, task, corridors, cell_to_id = build_scenario(
                        size=args.size, density=density, corridor_length=corridor_length,
                        n_agents=args.agents, seed=int(time.time()) & 0xFFFF,
                    )
                    rollout = Rollout(net, grid, task, corridors, cell_to_id, device, obs_spec, greedy=args.greedy)
                    selected_agent = 0
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    fps = min(60, fps + 1)
                    tick_ms_target = 1000 / max(1, fps)
                elif ev.key == pygame.K_MINUS:
                    fps = max(1, fps - 1)
                    tick_ms_target = 1000 / max(1, fps)
                elif ev.key == pygame.K_ESCAPE:
                    running = False
        now = time.time() * 1000
        if not paused and now - last_step >= tick_ms_target:
            rollout.step()
            last_step = now
        # Draw.
        screen.fill(BG)
        draw(screen, grid.obstacle_map, grid.positions, task.goals, cell)
        # HUD.
        pygame.draw.rect(screen, HUD_BG, (0, h_grid, w_grid, hud_h))
        throughput = rollout.total_arrivals / max(1, rollout.step_idx)
        lines = [
            f"step {rollout.step_idx:4d}   throughput {throughput:.3f}   arrivals {rollout.total_arrivals}",
            f"world {grid.h}x{grid.w}  agents {grid.n_agents}  seed {args.seed}  fps {fps}",
            f"[SPACE] pause  [R] reset  [V] obs panel  [+/-] speed  [click] select agent  [ESC] quit",
        ]
        for i, s in enumerate(lines):
            img = font.render(s, True, HUD_FG)
            screen.blit(img, (10, h_grid + 6 + i * 22))

        # Highlight selected agent.
        if show_panel:
            sr, sc = int(grid.positions[selected_agent, 0]), int(grid.positions[selected_agent, 1])
            pygame.draw.rect(screen, (0, 0, 0),
                             (sc * cell - 2, sr * cell - 2, cell + 4, cell + 4), width=2)
            # Obs panel background.
            pygame.draw.rect(screen, PANEL_BG, (w_grid, 0, panel_w, screen.get_height()))
            # Build obs for selected agent.
            rollout.builder.refresh_step_cache()
            spatial, scalars = rollout.builder.build(selected_agent)
            title = font.render(
                f"agent {selected_agent} obs channels  |  d(→goal)={scalars[2]:.2f}",
                True, (30, 30, 30),
            )
            screen.blit(title, (w_grid + 10, 8))
            draw_obs_panel(screen, spatial, (w_grid + 10, 30), font_small, cell=8)

        pygame.display.flip()
    pygame.quit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--corridor-length", type=int, default=5)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--greedy", action="store_true",
                   help="Use argmax actions instead of sampling from the policy distribution.")
    p.add_argument("--no-corridors", action="store_true",
                   help="Empty room, no obstacles at all (bypasses density/corridor-length).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
