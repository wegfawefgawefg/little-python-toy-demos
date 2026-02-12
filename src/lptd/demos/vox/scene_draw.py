from __future__ import annotations

from . import config
from .camera import scale_color
from .world import add_color_offset


def _depth_norm(depth: float) -> float:
    # Map camera depth to [0,1] for GL depth testing (near=0, far=1).
    z = depth / max(0.001, float(config.BLOCK_MAX_DIST))
    return max(0.0, min(1.0, z))


def _draw_flower_patch(prims, patch: dict, proj, factor: float, depth: float) -> None:
    sx, sy = proj
    stem_color = (0, 180, 0)
    stem_len = max(1, int(8 * factor))
    stem_w = max(1, int(1 * factor))

    for flower in patch["flowers"]:
        ox, oz = flower["offset"]
        fx = sx + ox * 6 * factor
        fy = sy + oz * 2 * factor

        prims.line(fx, fy, fx, fy - stem_len, stem_w, stem_color, depth=depth)

        petal_color = flower["color"]
        petal_radius = max(1, int(3 * factor))
        cx = int(fx)
        cy = int(fy - stem_len)
        prims.circle(cx, cy, petal_radius, petal_color, depth=depth)
        prims.triangle(
            (cx, cy - petal_radius),
            (cx - petal_radius, cy + petal_radius),
            (cx + petal_radius, cy + petal_radius),
            petal_color,
            depth=depth,
        )


def _draw_bunny(prims, entity: dict, proj, factor: float, depth: float) -> None:
    sx, sy = proj
    body_radius = max(2, int(5 * factor))
    ear_width = max(1, int(2 * factor))
    ear_height = max(2, int(6 * factor))

    vx, _, _ = entity.get("vel", (0.0, 0.0, 0.0))
    facing_left = vx < 0

    bunny_color = scale_color((200, 200, 200), 1)
    prims.circle(sx, sy, body_radius, bunny_color, depth=depth)

    if facing_left:
        ear1 = (
            (sx, sy - body_radius),
            (sx - ear_width, sy - body_radius - ear_height),
            (sx, sy - body_radius - ear_height),
        )
        ear2 = (
            (sx, sy - body_radius),
            (sx - ear_width, sy - body_radius - ear_height // 2),
            (sx, sy - body_radius - ear_height // 2),
        )
    else:
        ear1 = (
            (sx, sy - body_radius),
            (sx + ear_width, sy - body_radius - ear_height),
            (sx, sy - body_radius - ear_height),
        )
        ear2 = (
            (sx, sy - body_radius),
            (sx + ear_width, sy - body_radius - ear_height // 2),
            (sx, sy - body_radius - ear_height // 2),
        )

    prims.triangle(*ear1, bunny_color, depth=depth)
    prims.triangle(*ear2, bunny_color, depth=depth)

    eye_radius = max(1, int(0.5 * factor))
    eye_x = sx - body_radius // 2 if facing_left else sx + body_radius // 2
    eye_y = sy - body_radius // 4
    prims.circle(eye_x, eye_y, eye_radius, (0, 0, 0), depth=depth)


def draw_scene(prims, draw_list) -> None:
    prims.clear(config.SKY_COLOR)

    for item in draw_list:
        if item[1] == "block":
            depth, _, proj, size, color, brightness = item
            sx, sy = proj
            mod_color = scale_color(color, brightness)
            prims.rect_center(sx, sy, size, mod_color, depth=_depth_norm(depth))
        elif item[1] == "grass":
            depth, _, proj, factor, blades, color_offset = item
            sx, sy = proj
            brightness = max(0, 1 - (depth / config.BLOCK_MAX_DIST) ** 2)
            base_grass = add_color_offset(config.GRASS_COLOR, color_offset)
            final_grass_color = scale_color(base_grass, brightness)
            base_line = max(1, int((3 * factor) / 4))
            for length_mult, tip_offset in blades:
                blade_length = max(1, int(base_line * length_mult))
                tip_x = sx + tip_offset * factor
                tip_y = sy - blade_length
                prims.line(
                    sx,
                    sy,
                    tip_x,
                    tip_y,
                    1,
                    final_grass_color,
                    depth=_depth_norm(depth),
                )
        elif item[1] == "flower_patch":
            depth, _, proj, factor, patch = item
            _draw_flower_patch(prims, patch, proj, factor * 0.1, depth=_depth_norm(depth))
        elif item[1] == "bunny":
            depth, _, proj, factor, bunny = item
            _draw_bunny(prims, bunny, proj, factor * 0.1, depth=_depth_norm(depth))
