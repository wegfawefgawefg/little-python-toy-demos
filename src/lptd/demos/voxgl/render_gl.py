from __future__ import annotations

from .primitives import GLPrimitives
from .render_common import gather_frame_payload
from .scene_draw import draw_scene

_GL_PRIMS = GLPrimitives()


def draw_frame(pcx: int, pcy: int, pcz: int) -> None:
    block_instances, draw_list = gather_frame_payload(pcx, pcy, pcz, sort_items=False)
    draw_scene(_GL_PRIMS, draw_list, block_instances=block_instances)
