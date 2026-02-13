from __future__ import annotations

import math
import random

import noise
import numpy as np

from . import config, state

_FACE_NEIGHBORS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

_PACKED_BLOCK_DTYPE = np.dtype([("idx", np.uint16), ("bid", np.uint8)])

def _is_solid_world(wx: int, wy: int, wz: int) -> bool:
    if wy < 0:
        return True
    return (wx, wy, wz) in state.solid_blocks


def _is_surface_voxel(wx: int, wy: int, wz: int) -> bool:
    for dx, dy, dz in _FACE_NEIGHBORS:
        if not _is_solid_world(wx + dx, wy + dy, wz + dz):
            return True
    return False


def invalidate_chunk_draw_cache(key: tuple[int, int, int], include_neighbors: bool = False) -> None:
    state.chunk_draw_blocks.pop(key, None)
    if not include_neighbors:
        return
    cx, cy, cz = key
    for nk in (
        (cx - 1, cy, cz),
        (cx + 1, cy, cz),
        (cx, cy - 1, cz),
        (cx, cy + 1, cz),
        (cx, cy, cz - 1),
        (cx, cy, cz + 1),
    ):
        state.chunk_draw_blocks.pop(nk, None)


def rebuild_chunk_sprite_bases(key: tuple[int, int, int]) -> None:
    entries: list[tuple] = []
    for entity in state.chunk_entities.get(key, []):
        et = entity["type"]
        if et == "grass":
            px, py, pz = entity["pos"]
            entries.append(("grass", px, py, pz, entity["blades"], entity["color_offset"]))
        elif et == "flower_patch":
            bx, by, bz = entity["pos"]
            for flower in entity["flowers"]:
                ox, oz = flower["offset"]
                entries.append(("flower", bx + ox, by, bz + oz, flower["color"]))
    state.chunk_sprite_bases[key] = entries


def rebuild_chunk_draw_blocks(key: tuple[int, int, int]) -> None:
    chunk = state.chunks[key]
    non_air = np.argwhere(chunk != config.BLOCK_AIR)
    packed: list[tuple[int, int]] = []
    cx, cy, cz = key
    x0 = cx * config.CHUNK_SIZE
    y0 = cy * config.CHUNK_HEIGHT
    z0 = cz * config.CHUNK_SIZE
    for lx, ly, lz in non_air:
        i_lx = int(lx)
        i_ly = int(ly)
        i_lz = int(lz)
        wx = x0 + i_lx
        wy = y0 + i_ly
        wz = z0 + i_lz
        if _is_surface_voxel(wx, wy, wz):
            idx = i_lx | (i_lz << 4) | (i_ly << 8)
            packed.append((idx, int(chunk[i_lx, i_ly, i_lz])))
    if packed:
        state.chunk_draw_blocks[key] = np.array(packed, dtype=_PACKED_BLOCK_DTYPE)
    else:
        state.chunk_draw_blocks[key] = np.zeros((0,), dtype=_PACKED_BLOCK_DTYPE)


def generate_flower_patch(base_x: int, base_z: int, ground_h: int) -> dict:
    petal_colors = [(255, 255, 0), (255, 0, 0), (255, 192, 203)]
    num_flowers = random.randint(config.FLOWER_PATCH_MIN, config.FLOWER_PATCH_MAX)
    radius = random.uniform(config.FLOWER_PATCH_RADIUS_MIN, config.FLOWER_PATCH_RADIUS_MAX)
    flowers = []
    for _ in range(num_flowers):
        angle = random.uniform(0.0, math.tau)
        # sqrt(random()) gives a denser center and softer blob edge.
        r = radius * math.sqrt(random.random())
        offset_x = math.cos(angle) * r
        offset_z = math.sin(angle) * r
        color = random.choice(petal_colors)
        flowers.append({"offset": (offset_x, offset_z), "color": color})
    pos = (base_x + 0.5, ground_h + 1, base_z + 0.5)
    return {"type": "flower_patch", "pos": pos, "flowers": flowers}


def get_ground_height(chunk: np.ndarray, lx: int, lz: int) -> int:
    for y in range(config.CHUNK_HEIGHT - 1, -1, -1):
        if chunk[lx, y, lz] != config.BLOCK_AIR:
            return y
    return 0


def generate_bunny(cx: int, cy: int, cz: int, chunk: np.ndarray) -> dict:
    lx = random.randint(0, config.CHUNK_SIZE - 1)
    lz = random.randint(0, config.CHUNK_SIZE - 1)
    ground_h = get_ground_height(chunk, lx, lz)
    wx = cx * config.CHUNK_SIZE + lx
    wy = cy * config.CHUNK_HEIGHT + ground_h + 1
    wz = cz * config.CHUNK_SIZE + lz
    pos = [wx + 0.5, wy, wz + 0.5]
    hop_cycle = 0.5
    hop_timer = random.uniform(0, hop_cycle)
    vel = (random.uniform(-0.25, 0.25), 0.0, random.uniform(-0.25, 0.25))
    return {
        "type": "bunny",
        "pos": pos,
        "hop_timer": hop_timer,
        "hop_cycle": hop_cycle,
        "vel": vel,
    }


def generate_grass(x: int, z: int, h: int) -> dict:
    grass_pos = (x + 0.5, h + 1, z + 0.5)
    blades = []
    for _ in range(3):
        length_mult = random.uniform(0.75, 1.25)
        tip_offset = random.uniform(-1, 1)
        blades.append((length_mult, tip_offset))
    color_offset = (
        random.randint(-10, 10),
        random.randint(-10, 10),
        random.randint(-10, 10),
    )
    return {
        "type": "grass",
        "pos": grass_pos,
        "blades": blades,
        "color_offset": color_offset,
    }


def add_color_offset(base: tuple[int, int, int], offset: tuple[int, int, int]):
    return tuple(max(0, min(255, base[i] + offset[i])) for i in range(3))


def generate_chunk(cx: int, cy: int, cz: int) -> None:
    chunk_array = np.zeros(
        (config.CHUNK_SIZE, config.CHUNK_HEIGHT, config.CHUNK_SIZE), dtype=np.uint8
    )
    entity_list: list[dict] = []
    tree_requests: list[tuple[int, int, int]] = []

    x0 = cx * config.CHUNK_SIZE
    y0 = cy * config.CHUNK_HEIGHT
    z0 = cz * config.CHUNK_SIZE

    for lx in range(config.CHUNK_SIZE):
        for lz in range(config.CHUNK_SIZE):
            wx = x0 + lx
            wz = z0 + lz
            n = noise.pnoise2(
                wx * config.NOISE_SCALE,
                wz * config.NOISE_SCALE,
                octaves=config.NOISE_OCTAVES,
                persistence=config.NOISE_PERSISTENCE,
                lacunarity=config.NOISE_LACUNARITY,
                repeatx=1024,
                repeaty=1024,
                base=0,
            )
            h = int(config.HEIGHT_BASE + n * config.HEIGHT_AMP)

            local_top = y0 + config.CHUNK_HEIGHT - 1

            if y0 <= 0 <= local_top:
                ly = 0 - y0
                chunk_array[lx, ly, lz] = config.BLOCK_BEDROCK
                state.solid_blocks.add((wx, 0, wz))

            y_start = max(1, y0)
            y_end = min(h, local_top)
            if y_start <= y_end:
                for wy in range(y_start, y_end + 1):
                    ly = wy - y0
                    chunk_array[lx, ly, lz] = config.BLOCK_DIRT
                    state.solid_blocks.add((wx, wy, wz))

            if h < config.WATER_LEVEL:
                y_start = max(h + 1, y0)
                y_end = min(config.WATER_LEVEL, local_top)
                if y_start <= y_end:
                    for wy in range(y_start, y_end + 1):
                        ly = wy - y0
                        chunk_array[lx, ly, lz] = config.BLOCK_WATER
                        state.solid_blocks.add((wx, wy, wz))

            if cy == 0:
                if h >= config.WATER_LEVEL and random.random() < 0.25:
                    entity_list.append(generate_grass(wx, wz, h))
                if h >= config.WATER_LEVEL and random.random() < config.FLOWER_PATCH_CHANCE:
                    entity_list.append(generate_flower_patch(wx, wz, h))
                if h >= config.WATER_LEVEL and random.random() < 0.001:
                    tree_requests.append((wx, wz, h))

    key = (cx, cy, cz)
    state.chunks[key] = chunk_array

    if cy == 0 and random.random() < config.BUNNY_SPAWN_CHANCE:
        entity_list.append(generate_bunny(cx, cy, cz, chunk_array))

    state.chunk_entities[key] = entity_list
    rebuild_chunk_sprite_bases(key)

    for wx, wz, h in tree_requests:
        generate_tree_at(wx, wz, h)

    rebuild_chunk_draw_blocks(key)
    cx, cy, cz = key
    for nk in (
        (cx - 1, cy, cz),
        (cx + 1, cy, cz),
        (cx, cy - 1, cz),
        (cx, cy + 1, cz),
        (cx, cy, cz - 1),
        (cx, cy, cz + 1),
    ):
        invalidate_chunk_draw_cache(nk, include_neighbors=False)


def add_voxel(pos: tuple[int, int, int], color: tuple[int, int, int]) -> None:
    bid = config.COLOR_TO_ID.get(color)
    if bid is None:
        return
    x, y, z = pos
    cx = x // config.CHUNK_SIZE
    cy = y // config.CHUNK_HEIGHT
    cz = z // config.CHUNK_SIZE
    key = (cx, cy, cz)
    if key not in state.chunks:
        generate_chunk(cx, cy, cz)
    chunk = state.chunks[key]
    lx = x - cx * config.CHUNK_SIZE
    ly = y - cy * config.CHUNK_HEIGHT
    lz = z - cz * config.CHUNK_SIZE
    if (
        0 <= ly < config.CHUNK_HEIGHT
        and 0 <= lx < config.CHUNK_SIZE
        and 0 <= lz < config.CHUNK_SIZE
    ):
        prev = int(chunk[lx, ly, lz])
        if prev == bid:
            return
        chunk[lx, ly, lz] = bid
        state.solid_blocks.add(pos)
        invalidate_chunk_draw_cache(key, include_neighbors=True)


def draw_line(start, end, color):
    x0, y0, z0 = start
    x1, y1, z1 = end
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    steps = int(max(abs(dx), abs(dy), abs(dz)))
    positions = []
    if steps == 0:
        positions.append((int(round(x0)), int(round(y0)), int(round(z0))))
    for i in range(steps + 1):
        t = i / steps if steps else 0
        x = x0 + dx * t
        y = y0 + dy * t
        z = z0 + dz * t
        pos = (int(round(x)), int(round(y)), int(round(z)))
        if pos not in positions:
            positions.append(pos)
    for pos in positions:
        add_voxel(pos, color)
    return positions


def draw_sphere(center, radius, color):
    cx, cy, cz = center
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            for z in range(cz - radius, cz + radius + 1):
                if math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= radius:
                    add_voxel((x, y, z), color)


def generate_branches(start, direction, length, depth):
    if depth <= 0:
        tip = (int(round(start[0])), int(round(start[1])), int(round(start[2])))
        draw_sphere(tip, 2, config.ID_TO_COLOR[config.BLOCK_LEAVES])
        return
    end = (
        start[0] + direction[0] * length,
        start[1] + direction[1] * length,
        start[2] + direction[2] * length,
    )
    draw_line(start, end, config.ID_TO_COLOR[config.BLOCK_TRUNK])
    num_branches = random.randint(4, 6)
    for _ in range(num_branches):
        new_dir = (
            direction[0] + random.uniform(-0.5, 0.5),
            direction[1] + random.uniform(0.05, 0.25),
            direction[2] + random.uniform(-0.5, 0.5),
        )
        norm = math.sqrt(new_dir[0] ** 2 + new_dir[1] ** 2 + new_dir[2] ** 2)
        new_dir = (new_dir[0] / norm, new_dir[1] / norm, new_dir[2] / norm)
        new_length = length * random.uniform(0.7, 0.9)
        generate_branches(end, new_dir, new_length, depth - 1)


def generate_tree_at(x: int, z: int, ground_h: int) -> None:
    tree_base = (x, ground_h + 1, z)
    trunk_height = random.randint(2, 4)
    trunk_top = (x, ground_h + trunk_height, z)
    draw_line(tree_base, trunk_top, config.ID_TO_COLOR[config.BLOCK_TRUNK])
    initial_direction = (random.uniform(-0.3, 0.3), 1.0, random.uniform(-0.3, 0.3))
    norm = math.sqrt(initial_direction[0] ** 2 + initial_direction[1] ** 2 + initial_direction[2] ** 2)
    initial_direction = (
        initial_direction[0] / norm,
        initial_direction[1] / norm,
        initial_direction[2] / norm,
    )
    branch_length = random.uniform(4, 8)
    branch_depth = random.randint(3, 5)
    generate_branches(trunk_top, initial_direction, branch_length, branch_depth)


def ensure_chunks_around(pcx: int, pcy: int, pcz: int) -> None:
    vr = max(1, int(state.view_radius))
    cy_min = max(0, pcy - int(config.VIEW_RADIUS_Y_DOWN))
    cy_max = pcy + int(config.VIEW_RADIUS_Y_UP)
    for cx in range(pcx - vr, pcx + vr + 1):
        for cy in range(cy_min, cy_max + 1):
            for cz in range(pcz - vr, pcz + vr + 1):
                if (cx - pcx) ** 2 + (cz - pcz) ** 2 <= vr**2:
                    if (cx, cy, cz) not in state.chunks:
                        generate_chunk(cx, cy, cz)


def ensure_chunks_in_sight(pcx: int, pcy: int, pcz: int) -> None:
    """Prefetch chunks in front of the camera so looking around generates terrain."""
    vr = max(1, int(state.view_radius))
    cy_min = max(0, pcy - int(config.VIEW_RADIUS_Y_DOWN))
    cy_max = pcy + int(config.VIEW_RADIUS_Y_UP)
    cam_x = state.cam_pos[0]
    cam_z = state.cam_pos[2]
    fx = math.sin(state.cam_yaw)
    fz = math.cos(state.cam_yaw)
    half_angle = math.radians(float(config.FOV) * 0.5)
    # Slightly wider than visible frustum to avoid pop-in at the edge.
    cos_limit = math.cos(min(math.pi - 0.01, half_angle + 0.25))

    for cx in range(pcx - vr, pcx + vr + 1):
        for cz in range(pcz - vr, pcz + vr + 1):
            center_x = cx * config.CHUNK_SIZE + config.CHUNK_SIZE * 0.5
            center_z = cz * config.CHUNK_SIZE + config.CHUNK_SIZE * 0.5
            dx = center_x - cam_x
            dz = center_z - cam_z
            dist2 = dx * dx + dz * dz
            if dist2 <= 1e-6:
                for cy in range(cy_min, cy_max + 1):
                    key = (cx, cy, cz)
                    if key not in state.chunks:
                        generate_chunk(cx, cy, cz)
                continue
            inv_len = 1.0 / math.sqrt(dist2)
            ndot = (dx * fx + dz * fz) * inv_len
            if ndot >= cos_limit:
                for cy in range(cy_min, cy_max + 1):
                    key = (cx, cy, cz)
                    if key not in state.chunks:
                        generate_chunk(cx, cy, cz)
