from __future__ import annotations

from . import config, state
from .camera import scale_color
from .world import add_color_offset


def _flower_sprite_kind(color: tuple[int, int, int]) -> str:
    if color == (255, 255, 0):
        return "flower_yellow"
    if color == (255, 0, 0):
        return "flower_red"
    return "flower_pink"


def _depth_norm(depth: float) -> float:
    vr = max(1, int(state.view_radius))
    max_depth = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
    # Map camera depth to [0,1] for GL depth testing (near=0, far=1).
    z = depth / max(0.001, max_depth)
    return max(0.0, min(1.0, z))


def _max_depth() -> float:
    vr = max(1, int(state.view_radius))
    return max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))


def _depth_norm_biased(depth: float, toward_camera: float) -> float:
    # Small camera-space bias helps foliage sprites avoid z-fighting/ties with ground splats.
    return _depth_norm(max(0.101, depth - toward_camera))


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

    begin_quads = getattr(prims, "begin_quads", None)
    end_quads = getattr(prims, "end_quads", None)
    if callable(begin_quads):
        begin_quads()
    for item in draw_list:
        if item[1] != "block":
            continue
        depth, _, proj, size, color, brightness = item
        sx, sy = proj
        mod_color = scale_color(color, brightness)
        prims.rect_center(sx, sy, size, mod_color, depth=_depth_norm(depth))
    if callable(end_quads):
        end_quads()

    sprite_center = getattr(prims, "sprite_center", None)
    begin_sprites = getattr(prims, "begin_sprites", None)
    end_sprites = getattr(prims, "end_sprites", None)

    if callable(sprite_center) and callable(begin_sprites) and callable(end_sprites):
        max_depth = _max_depth()
        flower_size_scale = 1.2
        bunny_size_scale = flower_size_scale * 2.0
        sprite_instances: dict[str, list[tuple[float, float, int, tuple[int, int, int], float]]] = {
            "grass": [],
            "flower_yellow": [],
            "flower_red": [],
            "flower_pink": [],
            "bunny": [],
        }

        for item in draw_list:
            it = item[1]
            if it == "grass":
                depth, _, proj, factor, _blades, color_offset = item
                sx, sy = proj
                brightness = max(0, 1 - (depth / max_depth) ** 2)
                base_grass = add_color_offset(config.GRASS_COLOR, color_offset)
                final_grass_color = scale_color(base_grass, brightness)
                sprite_size = max(1, int(0.5 * factor))
                sprite_instances["grass"].append(
                    (
                        sx,
                        sy - max(1, int(0.8 * factor)),
                        sprite_size,
                        final_grass_color,
                        _depth_norm_biased(depth, toward_camera=0.2),
                    )
                )
            elif it == "flower_patch":
                _depth, _, _proj, _factor, _patch, flowers_projected = item
                for flower in flowers_projected:
                    fdepth = flower["depth"]
                    ffactor = flower["factor"]
                    brightness = max(0, 1 - (fdepth / max_depth) ** 2)
                    shade = max(0, min(255, int(brightness * 255)))
                    tint = (shade, shade, shade)
                    sprite_size = max(1, int(flower_size_scale * ffactor))
                    sprite_instances[_flower_sprite_kind(flower["color"])].append(
                        (
                            flower["sx"],
                            flower["sy"] - max(1, int(0.8 * ffactor)),
                            sprite_size,
                            tint,
                            _depth_norm_biased(fdepth, toward_camera=0.15),
                        )
                    )
            elif it == "bunny":
                depth, _, proj, factor, _bunny = item
                sx, sy = proj
                bunny_color = scale_color((200, 200, 200), 1)
                sprite_size = max(1, int(bunny_size_scale * factor))
                # Anchor bunny feet near the entity world position on ground.
                bunny_center_y = sy - (sprite_size * 0.48)
                sprite_instances["bunny"].append(
                    (
                        sx,
                        bunny_center_y,
                        sprite_size,
                        bunny_color,
                        _depth_norm_biased(depth, toward_camera=0.05),
                    )
                )

        for kind, instances in sprite_instances.items():
            if not instances:
                continue
            begin_sprites(kind)
            for sx, sy, size, color, z in instances:
                sprite_center(kind, sx, sy, size, color, depth=z)
            end_sprites()

    for item in draw_list:
        if item[1] == "grass":
            if callable(sprite_center):
                continue
            depth, _, proj, factor, blades, color_offset = item
            sx, sy = proj
            vr = max(1, int(state.view_radius))
            max_depth = max(float(config.BLOCK_MAX_DIST), float(vr * config.CHUNK_SIZE))
            brightness = max(0, 1 - (depth / max_depth) ** 2)
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
            if callable(sprite_center):
                continue
            depth, _, proj, factor, patch, _flowers_projected = item
            _draw_flower_patch(prims, patch, proj, factor * 0.1, depth=_depth_norm(depth))
        elif item[1] == "bunny":
            if callable(sprite_center):
                continue
            depth, _, proj, factor, bunny = item
            _draw_bunny(prims, bunny, proj, factor * 0.1, depth=_depth_norm(depth))
