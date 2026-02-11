from __future__ import annotations

import math

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
