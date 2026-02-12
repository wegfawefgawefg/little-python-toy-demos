from __future__ import annotations

import math

from . import config, state


def colliding_box(pos: tuple[float, float, float]) -> bool:
    x_min = pos[0] - config.PLAYER_HALF_WIDTH
    x_max = pos[0] + config.PLAYER_HALF_WIDTH
    y_min = pos[1]
    y_max = pos[1] + config.PLAYER_HEIGHT
    z_min = pos[2] - config.PLAYER_HALF_WIDTH
    z_max = pos[2] + config.PLAYER_HALF_WIDTH
    for bx in range(int(math.floor(x_min)), int(math.floor(x_max)) + 1):
        for by in range(int(math.floor(y_min)), int(math.floor(y_max)) + 1):
            for bz in range(int(math.floor(z_min)), int(math.floor(z_max)) + 1):
                if (bx, by, bz) in state.solid_blocks:
                    return True
    return False


def move_axis_tile(pos: list[float], delta: float, axis: int) -> float:
    sign = 1 if delta > 0 else -1 if delta < 0 else 0
    moved = 0.0
    while abs(moved) < abs(delta):
        step = sign * min(1, abs(delta) - abs(moved))
        trial = pos[axis] + step
        new_pos = pos.copy()
        new_pos[axis] = trial
        if colliding_box((new_pos[0], new_pos[1], new_pos[2])):
            break
        pos[axis] = trial
        moved += step
    return pos[axis]

