from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_GREATER,
    GL_LINES,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_TRIANGLE_FAN,
    GL_UNSIGNED_BYTE,
    glBegin,
    glAlphaFunc,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3ub,
    glEnable,
    glEnd,
    glDisable,
    glGenTextures,
    glLineWidth,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
)

_ASSET_DIR = Path(__file__).with_name("assets")
_SPRITE_FILES = {
    "grass": "grass_sprite.png",
    "bunny": "bunny_sprite.png",
    "flower_yellow": "flower_yellow.png",
    "flower_red": "flower_red.png",
    "flower_pink": "flower_pink.png",
}


def _load_sprite_surface(kind: str) -> pygame.Surface:
    filename = _SPRITE_FILES.get(kind)
    if filename is None:
        raise ValueError(f"unknown sprite kind: {kind}")
    return pygame.image.load(str(_ASSET_DIR / filename)).convert_alpha()


class SoftPrimitives:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self._sprite_cache: dict[str, pygame.Surface] = {}

    def clear(self, color: tuple[int, int, int]) -> None:
        self.surface.fill(color)

    def rect_center(
        self,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        rect = pygame.Rect(int(x - size / 2), int(y - size / 2), size, size)
        pygame.draw.rect(self.surface, color, rect)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        pygame.draw.line(self.surface, color, (x1, y1), (x2, y2), width)

    def triangle(self, p1, p2, p3, color: tuple[int, int, int], depth: float | None = None) -> None:
        pygame.draw.polygon(self.surface, color, [p1, p2, p3])

    def circle(
        self,
        x: float,
        y: float,
        radius: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        pygame.draw.circle(self.surface, color, (int(x), int(y)), radius)

    def begin_sprites(self, kind: str) -> None:
        _ = kind

    def end_sprites(self) -> None:
        pass

    def sprite_center(
        self,
        kind: str,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        _ = depth
        if size <= 0:
            return
        sprite = self._sprite_cache.get(kind)
        if sprite is None:
            sprite = _load_sprite_surface(kind)
            self._sprite_cache[kind] = sprite
        scaled = pygame.transform.scale(sprite, (size, size))
        tinted = scaled.copy()
        tinted.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.surface.blit(tinted, (int(x - size * 0.5), int(y - size * 0.5)))


class SoftZPrimitives:
    """Low-res software rasterizer with per-pixel z-buffer."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        self._sprite_cache: dict[str, pygame.Surface] = {}

    def clear(self, color: tuple[int, int, int]) -> None:
        self.color[:, :, 0] = color[0]
        self.color[:, :, 1] = color[1]
        self.color[:, :, 2] = color[2]
        self.depth.fill(np.inf)

    def present(self, surface: pygame.Surface) -> None:
        # pygame surfarray is (w, h, c), internal buffer is (h, w, c).
        pygame.surfarray.blit_array(surface, np.transpose(self.color, (1, 0, 2)))

    def _write_mask(self, xx, yy, mask, depth: float | None, color: tuple[int, int, int]) -> None:
        if mask.size == 0:
            return
        xxv = xx[mask]
        yyv = yy[mask]
        if xxv.size == 0:
            return

        if depth is None:
            self.color[yyv, xxv, 0] = color[0]
            self.color[yyv, xxv, 1] = color[1]
            self.color[yyv, xxv, 2] = color[2]
            return

        d = float(depth)
        keep = d < self.depth[yyv, xxv]
        if not np.any(keep):
            return
        kx = xxv[keep]
        ky = yyv[keep]
        self.depth[ky, kx] = d
        self.color[ky, kx, 0] = color[0]
        self.color[ky, kx, 1] = color[1]
        self.color[ky, kx, 2] = color[2]

    def rect_center(
        self,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        half = size / 2.0
        x0 = max(0, int(math.floor(x - half)))
        y0 = max(0, int(math.floor(y - half)))
        x1 = min(self.width, int(math.ceil(x + half)))
        y1 = min(self.height, int(math.ceil(y + half)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = np.ones_like(xx, dtype=bool)
        self._write_mask(xx, yy, mask, depth, color)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        dx = x2 - x1
        dy = y2 - y1
        steps = int(max(abs(dx), abs(dy), 1))
        xs = np.rint(np.linspace(x1, x2, steps + 1)).astype(np.int32)
        ys = np.rint(np.linspace(y1, y2, steps + 1)).astype(np.int32)
        r = max(0, int(width // 2))
        for xo in range(-r, r + 1):
            for yo in range(-r, r + 1):
                xx = xs + xo
                yy = ys + yo
                valid = (xx >= 0) & (xx < self.width) & (yy >= 0) & (yy < self.height)
                if not np.any(valid):
                    continue
                self._write_mask(xx[valid], yy[valid], np.ones(np.count_nonzero(valid), dtype=bool), depth, color)

    def triangle(self, p1, p2, p3, color: tuple[int, int, int], depth: float | None = None) -> None:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        min_x = max(0, int(math.floor(min(x1, x2, x3))))
        max_x = min(self.width - 1, int(math.ceil(max(x1, x2, x3))))
        min_y = max(0, int(math.floor(min(y1, y2, y3))))
        max_y = min(self.height - 1, int(math.ceil(max(y1, y2, y3))))
        if min_x > max_x or min_y > max_y:
            return

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if den == 0:
            return
        w1 = ((y2 - y3) * (xx - x3) + (x3 - x2) * (yy - y3)) / den
        w2 = ((y3 - y1) * (xx - x3) + (x1 - x3) * (yy - y3)) / den
        w3 = 1.0 - w1 - w2
        mask = (w1 >= 0.0) & (w2 >= 0.0) & (w3 >= 0.0)
        self._write_mask(xx, yy, mask, depth, color)

    def circle(
        self,
        x: float,
        y: float,
        radius: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        r = int(max(0, radius))
        min_x = max(0, int(math.floor(x - r)))
        max_x = min(self.width - 1, int(math.ceil(x + r)))
        min_y = max(0, int(math.floor(y - r)))
        max_y = min(self.height - 1, int(math.ceil(y + r)))
        if min_x > max_x or min_y > max_y:
            return

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        mask = (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r
        self._write_mask(xx, yy, mask, depth, color)

    def begin_sprites(self, kind: str) -> None:
        _ = kind

    def end_sprites(self) -> None:
        pass

    def sprite_center(
        self,
        kind: str,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        if size <= 0:
            return
        sprite = self._sprite_cache.get(kind)
        if sprite is None:
            sprite = _load_sprite_surface(kind)
            self._sprite_cache[kind] = sprite
        scaled = pygame.transform.scale(sprite, (size, size))
        rgb = np.transpose(pygame.surfarray.array3d(scaled), (1, 0, 2)).astype(np.uint16)
        alpha = np.transpose(pygame.surfarray.array_alpha(scaled), (1, 0))
        if rgb.size == 0:
            return
        rgb[:, :, 0] = (rgb[:, :, 0] * color[0]) // 255
        rgb[:, :, 1] = (rgb[:, :, 1] * color[1]) // 255
        rgb[:, :, 2] = (rgb[:, :, 2] * color[2]) // 255
        rgb8 = rgb.astype(np.uint8)

        x0 = int(math.floor(x - size * 0.5))
        y0 = int(math.floor(y - size * 0.5))
        x1 = x0 + size
        y1 = y0 + size

        cx0 = max(0, x0)
        cy0 = max(0, y0)
        cx1 = min(self.width, x1)
        cy1 = min(self.height, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return

        sx0 = cx0 - x0
        sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0)
        sy1 = sy0 + (cy1 - cy0)

        src_rgb = rgb8[sy0:sy1, sx0:sx1]
        src_a = alpha[sy0:sy1, sx0:sx1]
        mask = src_a > 0
        if not np.any(mask):
            return

        dst_depth = self.depth[cy0:cy1, cx0:cx1]
        dst_color = self.color[cy0:cy1, cx0:cx1]
        if depth is None:
            dst_color[mask] = src_rgb[mask]
            return
        d = float(depth)
        keep = mask & (d < dst_depth)
        if not np.any(keep):
            return
        dst_depth[keep] = d
        dst_color[keep] = src_rgb[keep]


class GLPrimitives:
    def __init__(self) -> None:
        self._quad_batch_active = False
        self._sprite_textures: dict[str, int] = {}
        self._sprite_batch_kind: str | None = None

    def clear(self, color: tuple[int, int, int]) -> None:
        r, g, b = [c / 255.0 for c in color]
        glClearColor(r, g, b, 1.0)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def begin_quads(self) -> None:
        if self._quad_batch_active:
            return
        self._quad_batch_active = True
        glBegin(GL_QUADS)

    def end_quads(self) -> None:
        if not self._quad_batch_active:
            return
        glEnd()
        self._quad_batch_active = False

    def _ensure_sprite_texture(self, kind: str) -> int:
        tex = self._sprite_textures.get(kind)
        if tex is not None:
            return tex

        surf = _load_sprite_surface(kind)
        w, h = surf.get_size()
        pixels = pygame.image.tobytes(surf, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            w,
            h,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels,
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        self._sprite_textures[kind] = tex
        return tex

    def begin_sprites(self, kind: str) -> None:
        tex = self._ensure_sprite_texture(kind)
        if self._quad_batch_active:
            self.end_quads()
        if self._sprite_batch_kind == kind:
            return
        if self._sprite_batch_kind is not None:
            self.end_sprites()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Avoid depth writes from fully transparent texels.
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.01)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        self._sprite_batch_kind = kind

    def end_sprites(self) -> None:
        if self._sprite_batch_kind is None:
            return
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_BLEND)
        self._sprite_batch_kind = None

    def sprite_center(
        self,
        kind: str,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        if self._sprite_batch_kind != kind:
            self.begin_sprites(kind)
        half = size / 2
        z = 0.0 if depth is None else -depth
        glColor3ub(*color)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(x - half, y - half, z)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(x + half, y - half, z)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(x + half, y + half, z)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(x - half, y + half, z)

    def rect_center(
        self,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        half = size / 2
        glColor3ub(*color)
        # With our ortho setup, positive Z maps "closer" in depth after projection.
        # Negate so larger logical depth (farther) ends up farther in depth buffer.
        z = 0.0 if depth is None else -depth
        if self._quad_batch_active:
            glVertex3f(x - half, y - half, z)
            glVertex3f(x + half, y - half, z)
            glVertex3f(x + half, y + half, z)
            glVertex3f(x - half, y + half, z)
        else:
            glBegin(GL_QUADS)
            glVertex3f(x - half, y - half, z)
            glVertex3f(x + half, y - half, z)
            glVertex3f(x + half, y + half, z)
            glVertex3f(x - half, y + half, z)
            glEnd()

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        glColor3ub(*color)
        glLineWidth(width)
        z = 0.0 if depth is None else -depth
        glBegin(GL_LINES)
        glVertex3f(x1, y1, z)
        glVertex3f(x2, y2, z)
        glEnd()

    def triangle(self, p1, p2, p3, color: tuple[int, int, int], depth: float | None = None) -> None:
        glColor3ub(*color)
        z = 0.0 if depth is None else -depth
        glBegin(GL_TRIANGLES)
        glVertex3f(p1[0], p1[1], z)
        glVertex3f(p2[0], p2[1], z)
        glVertex3f(p3[0], p3[1], z)
        glEnd()

    def circle(
        self,
        x: float,
        y: float,
        radius: int,
        color: tuple[int, int, int],
        depth: float | None = None,
    ) -> None:
        # Approximate filled circle with a fan; segment count scales mildly with radius.
        segments = max(10, min(32, radius * 2))
        glColor3ub(*color)
        z = 0.0 if depth is None else -depth
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, y, z)
        for i in range(segments + 1):
            t = (2 * math.pi * i) / segments
            glVertex3f(x + math.cos(t) * radius, y + math.sin(t) * radius, z)
        glEnd()
