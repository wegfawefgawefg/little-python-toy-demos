from __future__ import annotations

import random

import pygame

from . import config
from .logic import game_tick
from .render import draw_state
from .state import State


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    clock = pygame.time.Clock()

    init_snake = [(config.GRID_WIDTH // 2, config.GRID_HEIGHT // 2)]
    state = State(
        snake=init_snake,
        direction=(1, 0),
        food=(
            random.randint(0, config.GRID_WIDTH - 1),
            random.randint(0, config.GRID_HEIGHT - 1),
        ),
        score=0,
        game_over=False,
    )

    while not state.game_over:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                state = state._replace(game_over=True)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                state = state._replace(game_over=True)

        state = game_tick(state, events)
        draw_state(screen, state)
        clock.tick(config.FPS)

    pygame.quit()
    print("Game Over! Score:", state.score)
