import pygame, sys, math, random, noise, numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

pygame.init()
# --- Render Setup ---
render_width, render_height = 240, 160  # Low-res virtual surface.
WIDTH, HEIGHT = render_width * 4, render_height * 4  # Upscaled window.
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

# hide mouse
pygame.mouse.set_visible(False)
# lock mouse to window
pygame.event.set_grab(True)

# Setup orthographic projection (2D drawing)
glViewport(0, 0, WIDTH, HEIGHT)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0, render_width, render_height, 0)  # y inverted to match pygame's coords.
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glDisable(GL_DEPTH_TEST)

# --- World Generation Parameters ---
CHUNK_SIZE = 16
CHUNK_HEIGHT = 32  # Maximum vertical blocks per chunk.
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
chunks = {}  # (cx, cz) -> numpy array of shape (CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE)
chunk_entities = {}  # (cx, cz) -> list of entities
solid_blocks = set()  # Set of world coords (x,y,z) for collision


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
            # Bedrock at y=0.
            if CHUNK_HEIGHT > 0:
                chunk_array[lx, 0, lz] = BLOCK_BEDROCK
                solid_blocks.add((wx, 0, wz))
            # Dirt from y=1 to h.
            for y in range(1, h + 1):
                if y < CHUNK_HEIGHT:
                    chunk_array[lx, y, lz] = BLOCK_DIRT
                    solid_blocks.add((wx, y, wz))
            # Water if ground below water_level.
            if h < water_level:
                for y in range(h + 1, water_level + 1):
                    if y < CHUNK_HEIGHT:
                        chunk_array[lx, y, lz] = BLOCK_WATER
                        solid_blocks.add((wx, y, wz))
            # Grass entity generation.
            if h >= water_level and random.random() < 0.25:
                entity_list.append(generate_grass(wx, wz, h))
            # Tree generation request.
            if h >= water_level and random.random() < 0.001:
                tree_requests.append((wx, wz, h))
    chunks[(cx, cz)] = chunk_array
    chunk_entities[(cx, cz)] = entity_list
    # Process tree requests.
    for wx, wz, h in tree_requests:
        generate_tree_at(wx, wz, h)


def add_voxel(pos, color):
    """Place a voxel at pos with the given color by updating the chunk array."""
    bid = color_to_id.get(color)
    if bid is None:
        return  # Unrecognized color/type.
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
fov = 120


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


# --- OpenGL Drawing Helpers ---
def draw_quad(x, y, size, color):
    r, g, b = color
    glColor3ub(r, g, b)
    half = size / 2.0
    glBegin(GL_QUADS)
    glVertex2f(x - half, y - half)
    glVertex2f(x + half, y - half)
    glVertex2f(x + half, y + half)
    glVertex2f(x - half, y + half)
    glEnd()


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

view_radius = 4  # Using Manhattan distance for view radius.

# --- Main Loop ---
while True:
    dt = clock.get_time() / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- Mouse look ---
    mx, my = pygame.mouse.get_rel()
    cam_yaw += mx * mouse_sensitivity
    cam_pitch -= my * mouse_sensitivity
    cam_pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, cam_pitch))

    # --- WASD Movement ---
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

    # --- Jumping & Gravity ---
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

    # --- Dynamic Chunk Generation ---
    pcx = int(math.floor(cam_pos[0] / CHUNK_SIZE))
    pcz = int(math.floor(cam_pos[2] / CHUNK_SIZE))
    for cx in range(pcx - view_radius, pcx + view_radius + 1):
        for cz in range(pcz - view_radius, pcz + view_radius + 1):
            if (cx - pcx) ** 2 + (cz - pcz) ** 2 <= view_radius**2:
                if (cx, cz) not in chunks:
                    generate_chunk(cx, cz)

    # --- OpenGL Clear ---
    r, g, b = [c / 255.0 for c in SKY_COLOR]
    glClearColor(r, g, b, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    combined_draw_list = []
    # --- Gather draw calls for blocks ---
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
                            # Culling: skip if off-screen.
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
            # --- Gather draw calls for grass entities ---
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
    # --- Sort draw calls by depth ---
    combined_draw_list.sort(key=lambda item: item[0], reverse=True)

    # --- OpenGL Drawing ---
    for item in combined_draw_list:
        if item[1] == "block":
            depth, _, proj, factor, color, brightness = item
            sx, sy = proj
            size = max(1, int(1.3 * factor))
            mod_color = scale_color(color, brightness)
            draw_quad(sx, sy, size, mod_color)
        elif item[1] == "grass":
            depth, _, proj, factor, blades, color_offset = item
            sx, sy = proj
            brightness = max(0, 1 - (depth / BLOCK_MAX_DIST) ** 2)
            base_grass = add_color_offset(GRASS_COLOR, color_offset)
            final_color = scale_color(base_grass, brightness)
            glColor3ub(*final_color)
            glLineWidth(1)
            glBegin(GL_LINES)
            for length_mult, tip_offset in blades:
                base_line = max(1, int((3 * factor) / 4))
                blade_length = max(1, int(base_line * length_mult))
                tip_x = sx + tip_offset * factor
                tip_y = sy - blade_length
                glVertex2f(sx, sy)
                glVertex2f(tip_x, tip_y)
            glEnd()

    pygame.display.flip()
    clock.tick(144)
