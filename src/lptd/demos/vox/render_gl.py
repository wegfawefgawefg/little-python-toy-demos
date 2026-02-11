from __future__ import annotations

from .primitives import GLPrimitives
from .render_common import gather_draw_list
from .scene_draw import draw_scene


def draw_frame(pcx: int, pcz: int) -> None:
    # For screen-space billboard squares, painter ordering is more correct than
    # depth-buffered quads with a single depth value per primitive.
    draw_scene(GLPrimitives(), gather_draw_list(pcx, pcz, sort_items=True))
