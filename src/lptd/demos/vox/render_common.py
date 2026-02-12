from __future__ import annotations

import math

from . import config, state
from .camera import (
    build_view_mat4,
    build_view_rotation_mat3,
    project_point_vec3,
    world_to_camera_vec3,
)
from .world import rebuild_chunk_draw_blocks
from lptd.linalg import Vec3


def _estimate_block_screen_size(
    cam: Vec3,
    sx: float,
    sy: float,
    factor: float,
    axis_x_cam: Vec3,
    axis_z_cam: Vec3,
    half_w: float,
    half_h: float,
) -> int:
    """Estimate pixel coverage from projected spacing of adjacent voxel centers."""

    def spacing_to_axis(axis: Vec3) -> float:
        nz = cam.z + axis.z
        if nz <= 0.1:
            return 0.0
        nf = config.FOV / nz
        nsx = (cam.x + axis.x) * nf + half_w
        nsy = -(cam.y + axis.y) * nf + half_h
        dx = nsx - sx
        dy = nsy - sy
        return math.sqrt(dx * dx + dy * dy)

    spacing = max(spacing_to_axis(axis_x_cam), spacing_to_axis(axis_z_cam))
    if spacing <= 0.0:
        spacing = 1.3 * factor
    # Slight overlap bias to avoid hole patterns on sloped terrain.
    size = int(math.ceil(spacing * 1.08 + 0.35))
    return max(1, min(64, size))


def gather_draw_list(pcx: int, pcz: int, sort_items: bool = True):
    combined = []
    half_w = state.render_w / 2
    half_h = state.render_h / 2
    chunk_half_xz = config.CHUNK_SIZE * 0.5
    chunk_half_y = config.CHUNK_HEIGHT * 0.5
    chunk_radius = math.sqrt(chunk_half_xz**2 + chunk_half_y**2 + chunk_half_xz**2)

    cam_view = Vec3(state.cam_pos[0], state.cam_pos[1] + config.PLAYER_HEIGHT, state.cam_pos[2])
    view_rot = build_view_rotation_mat3()
    view_m4 = build_view_mat4()
    axis_x_cam = view_rot @ Vec3(1.0, 0.0, 0.0)
    axis_z_cam = view_rot @ Vec3(0.0, 0.0, 1.0)

    vr = max(1, int(state.view_radius))
    # Tie render-depth culling to runtime view radius so +/- has visible effect.
    max_dist = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
    for cx in range(pcx - vr, pcx + vr + 1):
        for cz in range(pcz - vr, pcz + vr + 1):
            key = (cx, cz)
            chunk_center_x = cx * config.CHUNK_SIZE + chunk_half_xz
            chunk_center_y = chunk_half_y
            chunk_center_z = cz * config.CHUNK_SIZE + chunk_half_xz
            chunk_cam = view_m4.transform_point(Vec3(chunk_center_x, chunk_center_y, chunk_center_z))
            cx1 = chunk_cam.x
            cy1 = chunk_cam.y
            cz2 = chunk_cam.z

            # Chunk-level frustum culling: conservative sphere test.
            if cz2 + chunk_radius <= 0.1:
                continue
            if cz2 - chunk_radius >= max_dist:
                continue
            z_for_fov = max(0.1, cz2)
            x_limit = (half_w * z_for_fov / config.FOV) + chunk_radius
            y_limit = (half_h * z_for_fov / config.FOV) + chunk_radius
            if abs(cx1) > x_limit or abs(cy1) > y_limit:
                continue

            if key in state.chunks:
                blocks = state.chunk_draw_blocks.get(key)
                if blocks is None:
                    rebuild_chunk_draw_blocks(key)
                    blocks = state.chunk_draw_blocks.get(key, [])

                x0 = cx * config.CHUNK_SIZE
                z0 = cz * config.CHUNK_SIZE
                for lx, ly, lz, bid in blocks:
                    world = Vec3(x0 + lx + 0.5, ly + 0.5, z0 + lz + 0.5)
                    cam = world_to_camera_vec3(world, view_rot, cam_view)

                    if cam.z <= 0.1 or cam.z > max_dist:
                        continue

                    proj, factor = project_point_vec3(cam)
                    if proj is None:
                        continue
                    sx = proj.x
                    sy = proj.y

                    if (
                        sx < -state.render_w
                        or sx > state.render_w * 2
                        or sy < -state.render_h
                        or sy > state.render_h * 2
                    ):
                        continue

                    brightness = max(0, 1 - (cam.z / max_dist) ** 2)
                    block_px_size = _estimate_block_screen_size(
                        cam,
                        sx,
                        sy,
                        factor,
                        axis_x_cam,
                        axis_z_cam,
                        half_w,
                        half_h,
                    )
                    combined.append(
                        (
                            cam.z,
                            "block",
                            (sx, sy),
                            block_px_size,
                            config.ID_TO_COLOR[bid],
                            brightness,
                        )
                    )

            if key in state.chunk_entities:
                for entity in state.chunk_entities[key]:
                    pos = entity["pos"]
                    cam = world_to_camera_vec3(Vec3(pos[0], pos[1], pos[2]), view_rot, cam_view)

                    if cam.z <= 0.1 or cam.z > max_dist:
                        continue

                    proj, factor = project_point_vec3(cam)
                    if proj is None:
                        continue
                    sx = proj.x
                    sy = proj.y
                    if (
                        sx < -state.render_w
                        or sx > state.render_w * 2
                        or sy < -state.render_h
                        or sy > state.render_h * 2
                    ):
                        continue
                    proj = (sx, sy)
                    depth = cam.z
                    et = entity["type"]
                    if et == "grass":
                        combined.append(
                            (depth, "grass", proj, factor, entity["blades"], entity["color_offset"])
                        )
                    elif et == "flower_patch":
                        combined.append((depth, "flower_patch", proj, factor, entity))
                    elif et == "bunny":
                        combined.append((depth, "bunny", proj, factor, entity))

    if sort_items:
        combined.sort(key=lambda item: item[0], reverse=True)
    return combined
