from __future__ import annotations

import pygame

from .primitives import SoftPrimitives
from .render_common import gather_draw_list
from .scene_draw import draw_scene


def draw_frame(render_surf: pygame.Surface, pcx: int, pcz: int) -> None:
    draw_scene(SoftPrimitives(render_surf), gather_draw_list(pcx, pcz, sort_items=True))
