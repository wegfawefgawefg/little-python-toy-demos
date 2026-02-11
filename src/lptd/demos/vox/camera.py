from __future__ import annotations

import math

from . import config, state


def project_point(point):
    x, y, z = point
    if z <= 0.1:
        return None, None
    factor = config.FOV / z
    sx = x * factor + state.render_w / 2
    sy = -y * factor + state.render_h / 2
    return (sx, sy), factor


def world_to_camera(point):
    cam_view = (state.cam_pos[0], state.cam_pos[1] + config.PLAYER_HEIGHT, state.cam_pos[2])
    x = point[0] - cam_view[0]
    y = point[1] - cam_view[1]
    z = point[2] - cam_view[2]

    cos_y = math.cos(state.cam_yaw)
    sin_y = math.sin(state.cam_yaw)
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y

    cos_p = math.cos(state.cam_pitch)
    sin_p = math.sin(state.cam_pitch)
    y1 = y * cos_p - z1 * sin_p
    z2 = y * sin_p + z1 * cos_p
    return (x1, y1, z2)


def scale_color(color, brightness: float):
    return tuple(max(0, min(255, int(c * brightness))) for c in color)
