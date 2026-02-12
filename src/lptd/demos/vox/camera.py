from __future__ import annotations

import math

from . import config, state
from lptd.linalg import Mat3, Mat4, Vec2, Vec3


def focal_length_px() -> float:
    # True horizontal FOV in degrees -> focal length in pixels.
    half_w = state.render_w * 0.5
    half_angle = math.radians(float(config.FOV) * 0.5)
    return half_w / max(1e-6, math.tan(half_angle))


def project_point(point):
    x, y, z = point
    if z <= 0.1:
        return None, None
    factor = focal_length_px() / z
    sx = x * factor + state.render_w / 2
    sy = -y * factor + state.render_h / 2
    return (sx, sy), factor


def project_point_vec3(point: Vec3):
    if point.z <= 0.1:
        return None, None
    factor = focal_length_px() / point.z
    return Vec2(point.x * factor + state.render_w / 2, -point.y * factor + state.render_h / 2), factor


def build_view_rotation_mat3() -> Mat3:
    # Match the existing manual transform convention:
    # yaw uses a negative angle here, then pitch.
    return Mat3.rotate_x(state.cam_pitch) @ Mat3.rotate_y(-state.cam_yaw)


def build_view_mat4() -> Mat4:
    cam_y = state.cam_pos[1] + config.PLAYER_HEIGHT
    return (
        Mat4.rotate_x(state.cam_pitch)
        @ Mat4.rotate_y(-state.cam_yaw)
        @ Mat4.translate(-state.cam_pos[0], -cam_y, -state.cam_pos[2])
    )


def world_to_camera_vec3(world: Vec3, view_rot: Mat3, cam_view: Vec3) -> Vec3:
    return view_rot @ (world - cam_view)


def world_to_camera(point):
    view_rot = build_view_rotation_mat3()
    cam_view = Vec3(state.cam_pos[0], state.cam_pos[1] + config.PLAYER_HEIGHT, state.cam_pos[2])
    v = world_to_camera_vec3(Vec3(point[0], point[1], point[2]), view_rot, cam_view)
    return (v.x, v.y, v.z)


def scale_color(color, brightness: float):
    return tuple(max(0, min(255, int(c * brightness))) for c in color)
