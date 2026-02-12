from __future__ import annotations

import math

from . import config, state
from .world import rebuild_chunk_draw_blocks


def gather_draw_list(pcx: int, pcz: int, sort_items: bool = True):
    combined = []
    half_w = state.render_w / 2
    half_h = state.render_h / 2
    chunk_half_xz = config.CHUNK_SIZE * 0.5
    chunk_half_y = config.CHUNK_HEIGHT * 0.5
    chunk_radius = math.sqrt(chunk_half_xz**2 + chunk_half_y**2 + chunk_half_xz**2)

    cam_x = state.cam_pos[0]
    cam_y = state.cam_pos[1] + config.PLAYER_HEIGHT
    cam_z = state.cam_pos[2]

    cos_y = math.cos(state.cam_yaw)
    sin_y = math.sin(state.cam_yaw)
    cos_p = math.cos(state.cam_pitch)
    sin_p = math.sin(state.cam_pitch)

    vr = max(1, int(state.view_radius))
    # Tie render-depth culling to runtime view radius so +/- has visible effect.
    max_dist = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
    for cx in range(pcx - vr, pcx + vr + 1):
        for cz in range(pcz - vr, pcz + vr + 1):
            key = (cx, cz)
            chunk_center_x = cx * config.CHUNK_SIZE + chunk_half_xz
            chunk_center_y = chunk_half_y
            chunk_center_z = cz * config.CHUNK_SIZE + chunk_half_xz
            rx = chunk_center_x - cam_x
            ry = chunk_center_y - cam_y
            rz = chunk_center_z - cam_z
            cx1 = rx * cos_y - rz * sin_y
            cz1 = rx * sin_y + rz * cos_y
            cy1 = ry * cos_p - cz1 * sin_p
            cz2 = ry * sin_p + cz1 * cos_p

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
                    wx = x0 + lx + 0.5
                    wy = ly + 0.5
                    wz = z0 + lz + 0.5

                    x = wx - cam_x
                    y = wy - cam_y
                    z = wz - cam_z

                    x1 = x * cos_y - z * sin_y
                    z1 = x * sin_y + z * cos_y
                    y1 = y * cos_p - z1 * sin_p
                    z2 = y * sin_p + z1 * cos_p

                    if z2 <= 0.1 or z2 > max_dist:
                        continue

                    factor = config.FOV / z2
                    sx = x1 * factor + half_w
                    sy = -y1 * factor + half_h

                    if (
                        sx < -state.render_w
                        or sx > state.render_w * 2
                        or sy < -state.render_h
                        or sy > state.render_h * 2
                    ):
                        continue

                    brightness = max(0, 1 - (z2 / max_dist) ** 2)
                    combined.append(
                        (
                            z2,
                            "block",
                            (sx, sy),
                            factor,
                            config.ID_TO_COLOR[bid],
                            brightness,
                        )
                    )

            if key in state.chunk_entities:
                for entity in state.chunk_entities[key]:
                    pos = entity["pos"]
                    x = pos[0] - cam_x
                    y = pos[1] - cam_y
                    z = pos[2] - cam_z

                    x1 = x * cos_y - z * sin_y
                    z1 = x * sin_y + z * cos_y
                    y1 = y * cos_p - z1 * sin_p
                    z2 = y * sin_p + z1 * cos_p

                    if z2 <= 0.1 or z2 > max_dist:
                        continue

                    factor = config.FOV / z2
                    sx = x1 * factor + half_w
                    sy = -y1 * factor + half_h
                    if (
                        sx < -state.render_w
                        or sx > state.render_w * 2
                        or sy < -state.render_h
                        or sy > state.render_h * 2
                    ):
                        continue
                    proj = (sx, sy)
                    depth = z2
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
