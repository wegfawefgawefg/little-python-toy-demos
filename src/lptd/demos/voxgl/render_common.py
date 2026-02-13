from __future__ import annotations

import math
import numpy as np

from . import config, state
from .camera import (
    build_view_mat4,
    build_view_rotation_mat3,
    focal_length_px,
    project_point_vec3,
    world_to_camera_vec3,
)
from .world import rebuild_chunk_draw_blocks, rebuild_chunk_sprite_bases
from lptd.linalg import Vec3

BLOCK_INSTANCE_DTYPE = np.dtype(
    [
        ("ox", np.float32),
        ("oy", np.float32),
        ("oz", np.float32),
        ("idx", np.uint16),
        ("bid", np.uint8),
    ]
)


def _estimate_block_screen_size(
    cam: Vec3,
    sx: float,
    sy: float,
    factor: float,
    axis_x_cam: Vec3,
    axis_z_cam: Vec3,
    half_w: float,
    half_h: float,
    focal_px: float,
) -> int:
    """Estimate pixel coverage from projected spacing of adjacent voxel centers."""

    def spacing_to_axis(axis: Vec3) -> float:
        nz = cam.z + axis.z
        if nz <= 0.1:
            return 0.0
        nf = focal_px / nz
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


def gather_draw_list(pcx: int, pcy: int, pcz: int, sort_items: bool = True):
    combined = []
    half_w = state.render_w / 2
    half_h = state.render_h / 2
    chunk_half_xz = config.CHUNK_SIZE * 0.5
    chunk_half_y = config.CHUNK_HEIGHT * 0.5
    chunk_radius = math.sqrt(chunk_half_xz**2 + chunk_half_y**2 + chunk_half_xz**2)
    focal_px = focal_length_px()

    cam_view = Vec3(state.cam_pos[0], state.cam_pos[1] + config.PLAYER_HEIGHT, state.cam_pos[2])
    view_rot = build_view_rotation_mat3()
    view_m4 = build_view_mat4()
    axis_x_cam = view_rot @ Vec3(1.0, 0.0, 0.0)
    axis_z_cam = view_rot @ Vec3(0.0, 0.0, 1.0)

    vr = max(1, int(state.view_radius))
    # Tie render-depth culling to runtime view radius so +/- has visible effect.
    max_dist = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
    cy_min = max(0, pcy - int(config.VIEW_RADIUS_Y_DOWN))
    cy_max = pcy + int(config.VIEW_RADIUS_Y_UP)
    for cx in range(pcx - vr, pcx + vr + 1):
        for cy in range(cy_min, cy_max + 1):
            for cz in range(pcz - vr, pcz + vr + 1):
                key = (cx, cy, cz)
                chunk_center_x = cx * config.CHUNK_SIZE + chunk_half_xz
                chunk_center_y = cy * config.CHUNK_HEIGHT + chunk_half_y
                chunk_center_z = cz * config.CHUNK_SIZE + chunk_half_xz
                chunk_cam = view_m4.transform_point(
                    Vec3(chunk_center_x, chunk_center_y, chunk_center_z)
                )
                cx1 = chunk_cam.x
                cy1 = chunk_cam.y
                cz2 = chunk_cam.z

                # Chunk-level frustum culling: conservative sphere test.
                if cz2 + chunk_radius <= 0.1:
                    continue
                if cz2 - chunk_radius >= max_dist:
                    continue
                z_for_fov = max(0.1, cz2)
                x_limit = (half_w * z_for_fov / focal_px) + chunk_radius
                y_limit = (half_h * z_for_fov / focal_px) + chunk_radius
                if abs(cx1) > x_limit or abs(cy1) > y_limit:
                    continue

                if key in state.chunks:
                    blocks = state.chunk_draw_blocks.get(key)
                    if blocks is None:
                        rebuild_chunk_draw_blocks(key)
                        blocks = state.chunk_draw_blocks.get(key, [])

                    x0 = cx * config.CHUNK_SIZE
                    y0 = cy * config.CHUNK_HEIGHT
                    z0 = cz * config.CHUNK_SIZE
                    for rec in blocks:
                        idx = int(rec["idx"])
                        bid = int(rec["bid"])
                        lx = idx & 0xF
                        lz = (idx >> 4) & 0xF
                        ly = idx >> 8
                        world = Vec3(x0 + lx + 0.5, y0 + ly + 0.5, z0 + lz + 0.5)
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
                            focal_px,
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

                sprite_bases = state.chunk_sprite_bases.get(key)
                if sprite_bases is None and key in state.chunk_entities:
                    rebuild_chunk_sprite_bases(key)
                    sprite_bases = state.chunk_sprite_bases.get(key, [])
                if sprite_bases:
                    for entry in sprite_bases:
                        et = entry[0]
                        cam = world_to_camera_vec3(
                            Vec3(entry[1], entry[2], entry[3]), view_rot, cam_view
                        )
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
                        if et == "grass":
                            combined.append((cam.z, "grass", (sx, sy), factor, entry[4], entry[5]))
                        elif et == "flower":
                            combined.append((cam.z, "flower", (sx, sy), factor, entry[4]))

                # Bunnies can be dynamic; keep direct entity path.
                if key in state.chunk_entities:
                    for entity in state.chunk_entities[key]:
                        if entity["type"] != "bunny":
                            continue
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
                        combined.append((cam.z, "bunny", (sx, sy), factor, entity))

    if sort_items:
        combined.sort(key=lambda item: item[0], reverse=True)
    return combined


def gather_frame_payload(pcx: int, pcy: int, pcz: int, sort_items: bool = False):
    """Return (block_instances, sprite_draw_items) for GPU-projected block path."""
    combined = []
    block_chunks: list[np.ndarray] = []
    half_w = state.render_w / 2
    half_h = state.render_h / 2
    chunk_half_xz = config.CHUNK_SIZE * 0.5
    chunk_half_y = config.CHUNK_HEIGHT * 0.5
    chunk_radius = math.sqrt(chunk_half_xz**2 + chunk_half_y**2 + chunk_half_xz**2)
    focal_px = focal_length_px()

    cam_view = Vec3(state.cam_pos[0], state.cam_pos[1] + config.PLAYER_HEIGHT, state.cam_pos[2])
    view_rot = build_view_rotation_mat3()
    view_m4 = build_view_mat4()

    vr = max(1, int(state.view_radius))
    max_dist = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
    cy_min = max(0, pcy - int(config.VIEW_RADIUS_Y_DOWN))
    cy_max = pcy + int(config.VIEW_RADIUS_Y_UP)
    for cx in range(pcx - vr, pcx + vr + 1):
        for cy in range(cy_min, cy_max + 1):
            for cz in range(pcz - vr, pcz + vr + 1):
                key = (cx, cy, cz)
                chunk_center_x = cx * config.CHUNK_SIZE + chunk_half_xz
                chunk_center_y = cy * config.CHUNK_HEIGHT + chunk_half_y
                chunk_center_z = cz * config.CHUNK_SIZE + chunk_half_xz
                chunk_cam = view_m4.transform_point(
                    Vec3(chunk_center_x, chunk_center_y, chunk_center_z)
                )
                cx1 = chunk_cam.x
                cy1 = chunk_cam.y
                cz2 = chunk_cam.z

                if cz2 + chunk_radius <= 0.1:
                    continue
                if cz2 - chunk_radius >= max_dist:
                    continue
                z_for_fov = max(0.1, cz2)
                x_limit = (half_w * z_for_fov / focal_px) + chunk_radius
                y_limit = (half_h * z_for_fov / focal_px) + chunk_radius
                if abs(cx1) > x_limit or abs(cy1) > y_limit:
                    continue

                if key in state.chunks:
                    blocks = state.chunk_draw_blocks.get(key)
                    if blocks is None:
                        rebuild_chunk_draw_blocks(key)
                        blocks = state.chunk_draw_blocks.get(key)
                    if blocks is not None and len(blocks) > 0:
                        chunk_instances = np.empty(len(blocks), dtype=BLOCK_INSTANCE_DTYPE)
                        chunk_instances["ox"] = np.float32(cx * config.CHUNK_SIZE)
                        chunk_instances["oy"] = np.float32(cy * config.CHUNK_HEIGHT)
                        chunk_instances["oz"] = np.float32(cz * config.CHUNK_SIZE)
                        chunk_instances["idx"] = blocks["idx"]
                        chunk_instances["bid"] = blocks["bid"]
                        block_chunks.append(chunk_instances)

                sprite_bases = state.chunk_sprite_bases.get(key)
                if sprite_bases is None and key in state.chunk_entities:
                    rebuild_chunk_sprite_bases(key)
                    sprite_bases = state.chunk_sprite_bases.get(key, [])
                if sprite_bases:
                    for entry in sprite_bases:
                        et = entry[0]
                        cam = world_to_camera_vec3(
                            Vec3(entry[1], entry[2], entry[3]), view_rot, cam_view
                        )
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
                        if et == "grass":
                            combined.append((cam.z, "grass", (sx, sy), factor, entry[4], entry[5]))
                        elif et == "flower":
                            combined.append((cam.z, "flower", (sx, sy), factor, entry[4]))

                if key in state.chunk_entities:
                    for entity in state.chunk_entities[key]:
                        if entity["type"] != "bunny":
                            continue
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
                        combined.append((cam.z, "bunny", (sx, sy), factor, entity))

    if sort_items:
        combined.sort(key=lambda item: item[0], reverse=True)
    if block_chunks:
        block_instances = np.concatenate(block_chunks)
    else:
        block_instances = np.zeros((0,), dtype=BLOCK_INSTANCE_DTYPE)
    return block_instances, combined
