from __future__ import annotations

from collections import deque

from . import config
from .pathfinding import bfs


class Snake:
    def __init__(self, color, init_pos: tuple[int, int]):
        self.color = color
        self.body = deque([init_pos])
        self.direction = (1, 0)
        self.next_move: tuple[int, int] | None = None

    def head(self) -> tuple[int, int]:
        return self.body[0]

    def update_direction(self, apple: tuple[int, int], obstacles: set[tuple[int, int]]):
        obs = set(obstacles)
        if len(self.body) > 1:
            obs -= {self.body[-1]}

        path = bfs(self.head(), apple, obs)
        if path:
            self.next_move = path[0]
            return

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self.head()[0] + dx, self.head()[1] + dy
            if 0 <= nx < config.GRID_WIDTH and 0 <= ny < config.GRID_HEIGHT and (nx, ny) not in obs:
                self.next_move = (nx, ny)
                return

        self.next_move = self.head()  # no valid move

    def move(self, apple_eaten: bool):
        if not self.next_move:
            return
        self.body.appendleft(self.next_move)
        if not apple_eaten:
            self.body.pop()

