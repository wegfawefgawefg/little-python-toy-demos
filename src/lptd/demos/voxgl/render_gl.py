from __future__ import annotations

from .primitives import GLPrimitives
from .render_common import gather_frame_payload
from .scene_draw import draw_scene


def draw_frame(pcx: int, pcz: int) -> None:
    block_instances, draw_list = gather_frame_payload(pcx, pcz, sort_items=False)
    draw_scene(GLPrimitives(), draw_list, block_instances=block_instances)
