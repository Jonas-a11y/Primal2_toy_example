"""Test the visualizer can initialize and render one frame without opening a window."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
import torch

from primal2_toy.policy.network import PolicyNet
from primal2_toy.obs.builder import ObsSpec
from primal2_toy.eval.rollout import build_scenario, Rollout
from primal2_toy.eval.visualizer import draw, draw_obs_panel


def main():
    pygame.init()
    obs_spec = ObsSpec(fov=11, n_pred=3)
    net = PolicyNet(
        n_spatial_channels=obs_spec.num_spatial_channels,
        n_scalars=obs_spec.num_scalar_features, n_actions=5, fov=11,
    )
    grid, task, corridors, cell_to_id = build_scenario(15, 0.3, 5, 4, 42)
    dev = torch.device("cpu")
    rollout = Rollout(net, grid, task, corridors, cell_to_id, dev, obs_spec)
    surf = pygame.Surface((800, 800))
    draw(surf, grid.obstacle_map, grid.positions, task.goals, 32)
    rollout.builder.refresh_step_cache()
    spatial, scalars = rollout.builder.build(0)
    font = pygame.font.SysFont("Menlo", 11)
    draw_obs_panel(surf, spatial, (0, 0), font, cell=8)
    print("visualizer_smoke OK — spatial shape", spatial.shape)


if __name__ == "__main__":
    main()
