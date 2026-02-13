from __future__ import annotations

from . import config, state
from .camera import build_view_rotation_mat3
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


def draw_scene(prims, draw_list, block_instances=None) -> None:
    prims.clear(config.SKY_COLOR)

    draw_block_points = getattr(prims, "draw_block_points", None)
    if callable(draw_block_points) and block_instances is not None:
        view_rot = build_view_rotation_mat3()
        cam_y = state.cam_pos[1] + config.PLAYER_HEIGHT
        draw_block_points(
            block_instances,
            state.cam_pos[0],
            cam_y,
            state.cam_pos[2],
            view_rot.m,
            state.render_w,
            state.render_h,
            float(config.FOV),
            _max_depth(),
        )
    else:
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
        elif it == "flower":
            depth, _, proj, factor, color = item
            sx, sy = proj
            brightness = max(0, 1 - (depth / max_depth) ** 2)
            shade = max(0, min(255, int(brightness * 255)))
            tint = (shade, shade, shade)
            sprite_size = max(1, int(flower_size_scale * factor))
            sprite_instances[_flower_sprite_kind(color)].append(
                (
                    sx,
                    sy - max(1, int(0.8 * factor)),
                    sprite_size,
                    tint,
                    _depth_norm_biased(depth, toward_camera=0.15),
                )
            )
        elif it == "bunny":
            depth, _, proj, factor, _bunny = item
            sx, sy = proj
            bunny_color = scale_color((200, 200, 200), 1)
            sprite_size = max(1, int(bunny_size_scale * factor))
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
        prims.begin_sprites(kind)
        for sx, sy, size, color, z in instances:
            prims.sprite_center(kind, sx, sy, size, color, depth=z)
        prims.end_sprites()
