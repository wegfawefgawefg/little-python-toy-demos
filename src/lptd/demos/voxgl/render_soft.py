from __future__ import annotations

import pygame

from .primitives import SoftZPrimitives
from .render_common import gather_draw_list
from .scene_draw import draw_scene

_zprims: SoftZPrimitives | None = None


def draw_frame(render_surf: pygame.Surface, pcx: int, pcz: int) -> None:
    global _zprims
    w, h = render_surf.get_size()
    if _zprims is None or _zprims.width != w or _zprims.height != h:
        _zprims = SoftZPrimitives(w, h)

    draw_scene(_zprims, gather_draw_list(pcx, pcz, sort_items=False))
    _zprims.present(render_surf)
