from __future__ import annotations

import pygame

from . import config
from .state import State


def draw_state(screen: pygame.Surface, state: State) -> None:
    screen.fill((0, 0, 0))

    for x, y in state.snake:
        rect = pygame.Rect(x * config.BLOCK, y * config.BLOCK, config.BLOCK, config.BLOCK)
        pygame.draw.rect(screen, (0, 255, 0), rect)

    fx, fy = state.food
    food_rect = pygame.Rect(fx * config.BLOCK, fy * config.BLOCK, config.BLOCK, config.BLOCK)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)

    pygame.display.flip()

