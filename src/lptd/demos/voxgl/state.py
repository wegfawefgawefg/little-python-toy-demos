from __future__ import annotations

chunks: dict[tuple[int, int], object] = {}
chunk_entities: dict[tuple[int, int], list[dict]] = {}
solid_blocks: set[tuple[int, int, int]] = set()
# Cached non-air voxels per chunk: (lx, ly, lz, block_id)
chunk_draw_blocks: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
# Cached static sprite bases per chunk.
# Entries:
#   ("grass", wx, wy, wz, blades, color_offset)
#   ("flower", wx, wy, wz, color)
chunk_sprite_bases: dict[tuple[int, int], list[tuple]] = {}

cam_pos = [0.0, 40.0, 0.0]
cam_yaw = 0.0
cam_pitch = 0.0

v_y = 0.0
grounded = False

# Internal scene render resolution (set at startup from CLI).
render_w = 240
render_h = 160

# Runtime-adjustable chunk view radius.
view_radius = 1
