from __future__ import annotations

import random
import sys

import pygame

from . import config
from .snake import Snake


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(config.SCREEN_SIZE)
    clock = pygame.time.Clock()

    snake1 = Snake(config.GREEN, (5, 5))
    snake2 = Snake(config.BLUE, (config.GRID_WIDTH - 6, config.GRID_HEIGHT - 6))
    snakes = [snake1, snake2]

    def new_apple():
        while True:
            pos = (
                random.randint(0, config.GRID_WIDTH - 1),
                random.randint(0, config.GRID_HEIGHT - 1),
            )
            if not any(pos in s.body for s in snakes):
                return pos

    apple = new_apple()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                sys.exit()

        obstacles: set[tuple[int, int]] = set()
        for s in snakes:
            obstacles.update(s.body)

        for s in snakes:
            other_obs = obstacles - {s.body[-1]}
            s.update_direction(apple, other_obs)

        # Conflict resolution: if both snakes plan the same next cell,
        # let snake2 try an alternate.
        if snake1.next_move == snake2.next_move:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                alt = (snake2.head()[0] + dx, snake2.head()[1] + dy)
                if (
                    0 <= alt[0] < config.GRID_WIDTH
                    and 0 <= alt[1] < config.GRID_HEIGHT
                    and alt not in obstacles
                ):
                    snake2.next_move = alt
                    break
            else:
                snake2.next_move = snake2.head()

        apple_eaten = [False, False]
        for idx, s in enumerate(snakes):
            if s.next_move == apple:
                apple_eaten[idx] = True

        for idx, s in enumerate(snakes):
            s.move(apple_eaten[idx])

        if any(apple_eaten):
            apple = new_apple()

        screen.fill(config.BLACK)
        pygame.draw.rect(
            screen,
            config.RED,
            (apple[0] * config.CELL_SIZE, apple[1] * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE),
        )
        for s in snakes:
            for seg in s.body:
                pygame.draw.rect(
                    screen,
                    s.color,
                    (seg[0] * config.CELL_SIZE, seg[1] * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE),
                )

        pygame.display.flip()
        clock.tick(config.FPS)
