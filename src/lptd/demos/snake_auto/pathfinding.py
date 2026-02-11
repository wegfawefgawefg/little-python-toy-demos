from __future__ import annotations

from collections import deque

from . import config


def bfs(start: tuple[int, int], goal: tuple[int, int], obstacles: set[tuple[int, int]]):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == goal:
            return path[1:]  # exclude starting cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < config.GRID_WIDTH and 0 <= ny < config.GRID_HEIGHT:
                if (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])
    return None

