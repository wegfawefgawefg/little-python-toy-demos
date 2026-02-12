from __future__ import annotations

import math

import numpy as np
import pygame
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINES,
    GL_QUADS,
    GL_TRIANGLES,
    GL_TRIANGLE_FAN,
    glBegin,
    glClear,
    glClearColor,
    glColor3ub,
    glDisable,
    glEnable,
    glEnd,
    glLineWidth,
    glVertex3f,
)


class SoftPrimitives:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface

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


class SoftZPrimitives:
    """Low-res software rasterizer with per-pixel z-buffer."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.full((self.height, self.width), np.inf, dtype=np.float32)

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


class GLPrimitives:
    def clear(self, color: tuple[int, int, int]) -> None:
        r, g, b = [c / 255.0 for c in color]
        glClearColor(r, g, b, 1.0)
        # Billboard/sprite-style quads are painter-sorted; disable depth test to
        # avoid incorrect occlusion from single-depth screen-space quads.
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)

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
