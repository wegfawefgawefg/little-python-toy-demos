import pygame, sys, math, random, noise, numpy as np
from pygame.locals import *

pygame.init()
# Render surface at low resolution.
render_width, render_height = 240, 160
# Upscaled window.
WIDTH, HEIGHT = render_width * 4, render_height * 4
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
render_surf = pygame.Surface((render_width, render_height))

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

# --- Projection Parameters ---
fov = 120

# --- World Generation Parameters ---
CHUNK_SIZE = 16
CHUNK_HEIGHT = 32  # maximum vertical blocks per chunk
water_level = 4
noise_scale = 0.1
noise_octaves = 4
noise_persistence = 0.5
noise_lacunarity = 2.0
height_base = 5
height_amp = 3

# --- Rendering Tweaks ---
BLOCK_MAX_DIST = 16

# --- Block Type Definitions ---
BLOCK_AIR = 0
BLOCK_BEDROCK = 1
BLOCK_DIRT = 2
BLOCK_WATER = 3
BLOCK_TRUNK = 4
BLOCK_LEAVES = 5

id_to_color = {
    BLOCK_BEDROCK: (80, 80, 80),
    BLOCK_DIRT: (120, 90, 60),
    BLOCK_WATER: (0, 0, 180),
    BLOCK_TRUNK: (80, 50, 40),
    BLOCK_LEAVES: (40, 80, 40),
}

color_to_id = {v: k for k, v in id_to_color.items()}

SKY_COLOR = (135, 206, 235)
GRASS_COLOR = (34, 139, 34)

# Global dictionaries for chunks and entities.
# chunks: (cx, cz) -> numpy array of shape (CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE)
chunks = {}
chunk_entities = {}  # (cx, cz) -> list of entities
solid_blocks = set()  # for collision; stores world coords (x,y,z)

# --- New Flower Patch Generation Functions ---


def generate_flower_patch(base_x, base_z, ground_h):
    """
    Generate a flower patch entity at (base_x, base_z) with ground height ground_h.
    The patch will contain several flowers arranged in a small cluster.
    Each flower is drawn with a green stem and three petal triangles.
    """
    petal_colors = [(255, 255, 0), (255, 0, 0), (255, 192, 203)]
    num_flowers = random.randint(5, 8)
    flowers = []
    for _ in range(num_flowers):
        offset_x = random.uniform(-2, 2)
        offset_z = random.uniform(-2, 2)
        if offset_x**2 + offset_z**2 > 4:
            continue
        color = random.choice(petal_colors)
        flowers.append({"offset": (offset_x, offset_z), "color": color})
    pos = (base_x + 0.5, ground_h + 1, base_z + 0.5)
    return {"type": "flower_patch", "pos": pos, "flowers": flowers}


# --- Bunny Generation ---


def get_ground_height(chunk, lx, lz):
    """Return the highest y (within CHUNK_HEIGHT) at (lx, lz) that is not air."""
    for y in range(CHUNK_HEIGHT - 1, -1, -1):
        if chunk[lx, y, lz] != BLOCK_AIR:
            return y
    return 0


def generate_bunny(cx, cz, chunk):
    """Generate a bunny entity in the chunk at a random ground position."""
    lx = random.randint(0, CHUNK_SIZE - 1)
    lz = random.randint(0, CHUNK_SIZE - 1)
    ground_h = get_ground_height(chunk, lx, lz)
    wx = cx * CHUNK_SIZE + lx
    wz = cz * CHUNK_SIZE + lz
    pos = [wx + 0.5, ground_h + 1, wz + 0.5]
    hop_cycle = 0.5  # seconds per hop cycle
    hop_timer = random.uniform(0, hop_cycle)
    # Add a velocity vector for wandering (x and z components).
    vel = (random.uniform(-0.5, 0.5), 0, random.uniform(-0.5, 0.5))
    return {
        "type": "bunny",
        "pos": pos,
        "hop_timer": hop_timer,
        "hop_cycle": hop_cycle,
        "vel": vel,
    }


# --- Helper Functions for World Generation ---


def generate_chunk(cx, cz):
    """Generate and store a chunk at (cx, cz) using a contiguous uint8 array."""
    chunk_array = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    entity_list = []
    tree_requests = []  # (world_x, world_z, ground_h)
    x0 = cx * CHUNK_SIZE
    z0 = cz * CHUNK_SIZE
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = x0 + lx
            wz = z0 + lz
            n = noise.pnoise2(
                wx * noise_scale,
                wz * noise_scale,
                octaves=noise_octaves,
                persistence=noise_persistence,
                lacunarity=noise_lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0,
            )
            h = int(height_base + n * height_amp)
            if CHUNK_HEIGHT > 0:
                chunk_array[lx, 0, lz] = BLOCK_BEDROCK
                solid_blocks.add((wx, 0, wz))
            for y in range(1, h + 1):
                if y < CHUNK_HEIGHT:
                    chunk_array[lx, y, lz] = BLOCK_DIRT
                    solid_blocks.add((wx, y, wz))
            if h < water_level:
                for y in range(h + 1, water_level + 1):
                    if y < CHUNK_HEIGHT:
                        chunk_array[lx, y, lz] = BLOCK_WATER
                        solid_blocks.add((wx, y, wz))
            if h >= water_level and random.random() < 0.25:
                entity_list.append(generate_grass(wx, wz, h))
            if h >= water_level and random.random() < 0.001:
                tree_requests.append((wx, wz, h))
    chunks[(cx, cz)] = chunk_array
    # One flower patch per chunk (patchy appearance)
    if random.random() < 0.05:
        lx = random.randint(0, CHUNK_SIZE - 1)
        lz = random.randint(0, CHUNK_SIZE - 1)
        wx = x0 + lx
        wz = z0 + lz
        ground_h = get_ground_height(chunk_array, lx, lz)
        entity_list.append(generate_flower_patch(wx, wz, ground_h))
    if random.random() < 0.2:
        entity_list.append(generate_bunny(cx, cz, chunk_array))
    chunk_entities[(cx, cz)] = entity_list
    for wx, wz, h in tree_requests:
        generate_tree_at(wx, wz, h)


def add_voxel(pos, color):
    """Place a voxel at pos with the given color by updating the chunk array."""
    bid = color_to_id.get(color)
    if bid is None:
        return
    x, y, z = pos
    cx = x // CHUNK_SIZE
    cz = z // CHUNK_SIZE
    key = (cx, cz)
    if key not in chunks:
        generate_chunk(cx, cz)
    chunk = chunks[key]
    lx = x - cx * CHUNK_SIZE
    lz = z - cz * CHUNK_SIZE
    if 0 <= y < CHUNK_HEIGHT and 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE:
        chunk[lx, y, lz] = bid
        solid_blocks.add(pos)


def draw_line(start, end, color):
    """Draw a line using DDA and fill blocks along it."""
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
    """Fill blocks within a sphere centered at center."""
    cx, cy, cz = center
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            for z in range(cz - radius, cz + radius + 1):
                if math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= radius:
                    add_voxel((x, y, z), color)


def generate_branches(start, direction, length, depth):
    """Recursively generate branches from start using a normalized direction."""
    if depth <= 0:
        tip = (int(round(start[0])), int(round(start[1])), int(round(start[2])))
        draw_sphere(tip, 2, id_to_color[BLOCK_LEAVES])
        return
    end = (
        start[0] + direction[0] * length,
        start[1] + direction[1] * length,
        start[2] + direction[2] * length,
    )
    draw_line(start, end, id_to_color[BLOCK_TRUNK])
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


def generate_tree_at(x, z, ground_h):
    """Generate a tree at the given world coordinates and ground height."""
    print(f"Tree at ({x},{z}): ground_h = {ground_h}")
    tree_base = (x, ground_h + 1, z)
    trunk_height = random.randint(2, 4)
    trunk_top = (x, ground_h + trunk_height, z)
    draw_line(tree_base, trunk_top, id_to_color[BLOCK_TRUNK])
    initial_direction = (random.uniform(-0.3, 0.3), 1.0, random.uniform(-0.3, 0.3))
    norm = math.sqrt(
        initial_direction[0] ** 2
        + initial_direction[1] ** 2
        + initial_direction[2] ** 2
    )
    initial_direction = (
        initial_direction[0] / norm,
        initial_direction[1] / norm,
        initial_direction[2] / norm,
    )
    branch_length = random.uniform(4, 8)
    branch_depth = random.randint(3, 5)
    generate_branches(trunk_top, initial_direction, branch_length, branch_depth)


# --- Grass Entity Generation ---


def generate_grass(x, z, h):
    """Generate a grass entity at block (x,z) with height h."""
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


def add_color_offset(base, offset):
    return tuple(max(0, min(255, base[i] + offset[i])) for i in range(3))


# --- Rendering / Camera / Physics Helpers ---


def colliding_box(pos):
    x_min = pos[0] - PLAYER_HALF_WIDTH
    x_max = pos[0] + PLAYER_HALF_WIDTH
    y_min = pos[1]
    y_max = pos[1] + PLAYER_HEIGHT
    z_min = pos[2] - PLAYER_HALF_WIDTH
    z_max = pos[2] + PLAYER_HALF_WIDTH
    for bx in range(int(math.floor(x_min)), int(math.floor(x_max)) + 1):
        for by in range(int(math.floor(y_min)), int(math.floor(y_max)) + 1):
            for bz in range(int(math.floor(z_min)), int(math.floor(z_max)) + 1):
                if (bx, by, bz) in solid_blocks:
                    return True
    return False


def move_axis_tile(pos, delta, axis):
    sign = 1 if delta > 0 else -1 if delta < 0 else 0
    moved = 0.0
    while abs(moved) < abs(delta):
        step = sign * min(1, abs(delta) - abs(moved))
        trial = pos[axis] + step
        new_pos = pos.copy()
        new_pos[axis] = trial
        if colliding_box(tuple(new_pos)):
            break
        pos[axis] = trial
        moved += step
    return pos[axis]


def project_point(point):
    x, y, z = point
    if z <= 0.1:
        return None, None
    factor = fov / z
    sx = x * factor + render_width / 2
    sy = -y * factor + render_height / 2
    return (sx, sy), factor


def world_to_camera(point):
    cam_view = (cam_pos[0], cam_pos[1] + PLAYER_HEIGHT, cam_pos[2])
    x = point[0] - cam_view[0]
    y = point[1] - cam_view[1]
    z = point[2] - cam_view[2]
    cos_y = math.cos(cam_yaw)
    sin_y = math.sin(cam_yaw)
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    cos_p = math.cos(cam_pitch)
    sin_p = math.sin(cam_pitch)
    y1 = y * cos_p - z1 * sin_p
    z2 = y * sin_p + z1 * cos_p
    return (x1, y1, z2)


def scale_color(color, brightness):
    return tuple(max(0, min(255, int(c * brightness))) for c in color)


# --- Flower Drawing Function ---


def draw_flower(patch, proj, factor):
    """
    Draw each flower in a patch.
    Each flower is drawn with a green stem and three petal triangles.
    """
    base_sx, base_sy = proj
    for flower in patch["flowers"]:
        off_x, off_z = flower["offset"]
        sx = base_sx + off_x * factor * 2
        sy = base_sy + off_z * factor * 2
        stem_length = max(1, int(4 * factor))
        stem_color = (0, 150, 0)
        pygame.draw.line(render_surf, stem_color, (sx, sy), (sx, sy - stem_length), 1)
        top_x, top_y = sx, sy - stem_length
        petal_size = max(1, int(3 * factor))
        flower_color = flower["color"]
        # Left petal
        points = [
            (top_x, top_y),
            (top_x - petal_size, top_y - petal_size),
            (top_x, top_y - int(petal_size * 0.7)),
        ]
        pygame.draw.polygon(render_surf, flower_color, points)
        # Right petal
        points = [
            (top_x, top_y),
            (top_x + petal_size, top_y - petal_size),
            (top_x, top_y - int(petal_size * 0.7)),
        ]
        pygame.draw.polygon(render_surf, flower_color, points)
        # Top petal
        points = [
            (top_x, top_y),
            (top_x - petal_size // 2, top_y - petal_size),
            (top_x + petal_size // 2, top_y - petal_size),
        ]
        pygame.draw.polygon(render_surf, flower_color, points)


# --- Bunny Drawing Function ---


def draw_bunny_entity(entity, proj, factor):
    """
    Draw a bunny using composite shapes.
    Its appearance changes based on its hop phase (up, down, or on the ground)
    and faces left or right based on its horizontal velocity.
    """
    t = entity["hop_timer"]
    cycle = entity["hop_cycle"]
    if t < cycle / 3:
        state = "ground"
    elif t < 2 * cycle / 3:
        state = "up"
    else:
        state = "down"
    # Compute vertical offset for hop effect.
    if state == "up":
        offset_y = -2 * factor
    elif state == "down":
        offset_y = 2 * factor
    else:
        offset_y = 0
    sx, sy = proj
    sy += offset_y
    vx = entity["vel"][0]
    facing = "left" if vx < 0 else "right"
    body_radius = max(1, int(3 * factor))
    body_color = scale_color((200, 200, 200), 1)
    pygame.draw.circle(render_surf, body_color, (int(sx), int(sy)), body_radius)
    ear_height = max(1, int(4 * factor))
    ear_width = max(1, int(2 * factor))
    if state == "up":
        if facing == "left":
            ear1 = [
                (sx - body_radius, sy - body_radius),
                (sx - body_radius - ear_width, sy - body_radius - ear_height),
                (sx - body_radius, sy - body_radius - ear_height),
            ]
            ear2 = [
                (sx, sy - body_radius),
                (sx - ear_width, sy - body_radius - ear_height),
                (sx, sy - body_radius - ear_height),
            ]
        else:
            ear1 = [
                (sx + body_radius, sy - body_radius),
                (sx + body_radius + ear_width, sy - body_radius - ear_height),
                (sx + body_radius, sy - body_radius - ear_height),
            ]
            ear2 = [
                (sx, sy - body_radius),
                (sx + ear_width, sy - body_radius - ear_height),
                (sx, sy - body_radius - ear_height),
            ]
    elif state == "down":
        if facing == "left":
            ear1 = [
                (sx - body_radius, sy - body_radius + 2),
                (sx - body_radius - ear_width, sy - body_radius + 2),
                (sx - body_radius, sy - body_radius + ear_height),
            ]
            ear2 = [
                (sx, sy - body_radius + 2),
                (sx - ear_width, sy - body_radius + 2),
                (sx, sy - body_radius + ear_height),
            ]
        else:
            ear1 = [
                (sx + body_radius, sy - body_radius + 2),
                (sx + body_radius + ear_width, sy - body_radius + 2),
                (sx + body_radius, sy - body_radius + ear_height),
            ]
            ear2 = [
                (sx, sy - body_radius + 2),
                (sx + ear_width, sy - body_radius + 2),
                (sx, sy - body_radius + ear_height),
            ]
    else:  # ground state
        if facing == "left":
            ear1 = [
                (sx - body_radius, sy - body_radius),
                (sx - body_radius - ear_width, sy - body_radius - ear_height // 2),
                (sx - body_radius, sy - body_radius - ear_height // 2),
            ]
            ear2 = [
                (sx, sy - body_radius),
                (sx - ear_width, sy - body_radius - ear_height // 2),
                (sx, sy - body_radius - ear_height // 2),
            ]
        else:
            ear1 = [
                (sx + body_radius, sy - body_radius),
                (sx + body_radius + ear_width, sy - body_radius - ear_height // 2),
                (sx + body_radius, sy - body_radius - ear_height // 2),
            ]
            ear2 = [
                (sx, sy - body_radius),
                (sx + ear_width, sy - body_radius - ear_height // 2),
                (sx, sy - body_radius - ear_height // 2),
            ]
    bunny_color = scale_color((200, 200, 200), 1)
    pygame.draw.polygon(render_surf, bunny_color, ear1)
    pygame.draw.polygon(render_surf, bunny_color, ear2)
    eye_radius = max(1, int(0.5 * factor))
    if facing == "left":
        eye_x = sx - body_radius // 2
    else:
        eye_x = sx + body_radius // 2
    eye_y = sy - body_radius // 4
    eye_color = (0, 0, 0)
    pygame.draw.circle(render_surf, eye_color, (int(eye_x), int(eye_y)), eye_radius)


# --- Player/Camera Settings ---
cam_pos = [0, 40, 0]
cam_yaw = 0
cam_pitch = 0
mouse_sensitivity = 0.003
PLAYER_HEIGHT = 2
PLAYER_HALF_WIDTH = 0.5
gravity = -20.0
jump_impulse = 7.0
v_y = 0.0
grounded = False
speed = 16.0

# Using Manhattan distance for view radius.
view_radius = 4

# --- Main Loop ---
while True:
    dt = clock.get_time() / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    # Mouse look.
    mx, my = pygame.mouse.get_rel()
    cam_yaw += mx * mouse_sensitivity
    cam_pitch -= my * mouse_sensitivity
    cam_pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, cam_pitch))
    # WASD movement.
    keys = pygame.key.get_pressed()
    forward_input = 0.0
    right_input = 0.0
    if keys[pygame.K_w]:
        forward_input += 1.0
    if keys[pygame.K_s]:
        forward_input -= 1.0
    if keys[pygame.K_d]:
        right_input += 1.0
    if keys[pygame.K_a]:
        right_input -= 1.0
    horizontal_forward = (math.sin(cam_yaw), 0, math.cos(cam_yaw))
    right_vec = (horizontal_forward[2], 0, -horizontal_forward[0])
    move_vector = (
        forward_input * horizontal_forward[0] + right_input * right_vec[0],
        0,
        forward_input * horizontal_forward[2] + right_input * right_vec[2],
    )
    mag = math.sqrt(move_vector[0] ** 2 + move_vector[2] ** 2)
    if mag > 0:
        move_vector = (move_vector[0] / mag, 0, move_vector[2] / mag)
    dx = move_vector[0] * speed * dt
    dz = move_vector[2] * speed * dt
    cam_pos[0] = move_axis_tile(cam_pos, dx, 0)
    cam_pos[2] = move_axis_tile(cam_pos, dz, 2)
    # Jumping.
    if keys[pygame.K_SPACE] and grounded:
        v_y = jump_impulse
        grounded = False
    v_y += gravity * dt
    dy = v_y * dt
    orig_y = cam_pos[1]
    new_y = move_axis_tile(cam_pos, dy, 1)
    if new_y == orig_y and dy < 0:
        v_y = 0
        grounded = True
    else:
        grounded = False
    # Dynamic Chunk Generation.
    pcx = int(math.floor(cam_pos[0] / CHUNK_SIZE))
    pcz = int(math.floor(cam_pos[2] / CHUNK_SIZE))
    for cx in range(pcx - view_radius, pcx + view_radius + 1):
        for cz in range(pcz - view_radius, pcz + view_radius + 1):
            if (cx - pcx) ** 2 + (cz - pcz) ** 2 <= view_radius**2:
                if (cx, cz) not in chunks:
                    generate_chunk(cx, cz)
    # Update bunny hop timers and wander.
    for ents in chunk_entities.values():
        for entity in ents:
            if entity["type"] == "bunny":
                entity["hop_timer"] = (entity["hop_timer"] + dt) % entity["hop_cycle"]
                vx, _, vz = entity["vel"]
                vx += random.uniform(-0.1, 0.1)
                vz += random.uniform(-0.1, 0.1)
                speed_val = math.sqrt(vx**2 + vz**2)
                max_speed = 1.0
                if speed_val > max_speed:
                    factor_speed = max_speed / speed_val
                    vx *= factor_speed
                    vz *= factor_speed
                entity["vel"] = (vx, 0, vz)
                entity["pos"][0] += vx * dt * 2
                entity["pos"][2] += vz * dt * 2
    render_surf.fill(SKY_COLOR)
    combined_draw_list = []
    # Render nearby chunks.
    for cx in range(pcx - view_radius, pcx + view_radius + 1):
        for cz in range(pcz - view_radius, pcz + view_radius + 1):
            key = (cx, cz)
            if key in chunks:
                chunk = chunks[key]
                x0 = cx * CHUNK_SIZE
                z0 = cz * CHUNK_SIZE
                for lx in range(CHUNK_SIZE):
                    for ly in range(CHUNK_HEIGHT):
                        for lz in range(CHUNK_SIZE):
                            bid = chunk[lx, ly, lz]
                            if bid == BLOCK_AIR:
                                continue
                            world_pos = (x0 + lx, ly, z0 + lz)
                            center = (
                                world_pos[0] + 0.5,
                                world_pos[1] + 0.5,
                                world_pos[2] + 0.5,
                            )
                            cam_space = world_to_camera(center)
                            if cam_space[2] <= 0.1 or cam_space[2] > BLOCK_MAX_DIST:
                                continue
                            proj, factor = project_point(cam_space)
                            if proj is None:
                                continue
                            sx, sy = proj
                            if (
                                sx < -render_width
                                or sx > render_width * 2
                                or sy < -render_height
                                or sy > render_height * 2
                            ):
                                continue
                            depth = cam_space[2]
                            brightness = max(0, 1 - (depth / BLOCK_MAX_DIST) ** 2)
                            combined_draw_list.append(
                                (
                                    depth,
                                    "block",
                                    proj,
                                    factor,
                                    id_to_color[bid],
                                    brightness,
                                )
                            )
            if key in chunk_entities:
                for entity in chunk_entities[key]:
                    if entity["type"] == "grass":
                        pos = entity["pos"]
                        cam_space = world_to_camera(pos)
                        if cam_space[2] <= 0.1 or cam_space[2] > BLOCK_MAX_DIST:
                            continue
                        proj, factor = project_point(cam_space)
                        if proj is None:
                            continue
                        depth = cam_space[2]
                        combined_draw_list.append(
                            (
                                depth,
                                "grass",
                                proj,
                                factor,
                                entity["blades"],
                                entity["color_offset"],
                            )
                        )
                    elif entity["type"] == "flower_patch":
                        pos = entity["pos"]
                        cam_space = world_to_camera(pos)
                        if cam_space[2] <= 0.1 or cam_space[2] > BLOCK_MAX_DIST:
                            continue
                        proj, factor = project_point(cam_space)
                        if proj is None:
                            continue
                        depth = cam_space[2]
                        combined_draw_list.append(
                            (depth, "flower_patch", proj, factor, entity)
                        )
                    elif entity["type"] == "bunny":
                        pos = entity["pos"]
                        cam_space = world_to_camera(pos)
                        if cam_space[2] <= 0.1 or cam_space[2] > BLOCK_MAX_DIST:
                            continue
                        proj, factor = project_point(cam_space)
                        if proj is None:
                            continue
                        depth = cam_space[2]
                        combined_draw_list.append(
                            (depth, "bunny", proj, factor, entity)
                        )
    combined_draw_list.sort(key=lambda item: item[0], reverse=True)
    for item in combined_draw_list:
        if item[1] == "block":
            depth, _, proj, factor, color, brightness = item
            sx, sy = proj
            size = max(1, int(1.3 * factor))
            mod_color = scale_color(color, brightness)
            rect = pygame.Rect(int(sx - size / 2), int(sy - size / 2), size, size)
            pygame.draw.rect(render_surf, mod_color, rect)
        elif item[1] == "grass":
            depth, _, proj, factor, blades, color_offset = item
            sx, sy = proj
            brightness = max(0, 1 - (depth / BLOCK_MAX_DIST) ** 2)
            base_grass = add_color_offset(GRASS_COLOR, color_offset)
            final_grass_color = scale_color(base_grass, brightness)
            base_line = max(1, int((3 * factor) / 4))
            for length_mult, tip_offset in blades:
                blade_length = max(1, int(base_line * length_mult))
                tip_x = sx + tip_offset * factor
                tip_y = sy - blade_length
                pygame.draw.line(
                    render_surf, final_grass_color, (sx, sy), (tip_x, tip_y), 1
                )
        elif item[1] == "flower_patch":
            depth, _, proj, factor, patch = item
            draw_flower(patch, proj, factor * 0.1)
        elif item[1] == "bunny":
            depth, _, proj, factor, bunny_entity = item
            draw_bunny_entity(bunny_entity, proj, factor * 0.1)
    scaled = pygame.transform.scale(render_surf, (WIDTH, HEIGHT))
    screen.blit(scaled, (0, 0))
    pygame.display.flip()
    clock.tick(144)
