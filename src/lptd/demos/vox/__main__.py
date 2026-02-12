from __future__ import annotations

import argparse
import math
import sys

import pygame

from . import config, state
from .hud import draw_status
from .player import update as update_player
from .primitives import GLPrimitives, SoftPrimitives
from .sim import update_entities
from .world import ensure_chunks_around


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vox", add_help=True)
    parser.add_argument(
        "--renderer",
        choices=("soft", "gl"),
        default="soft",
        help="Rendering backend (soft=pygame surface, gl=OpenGL).",
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help="Internal scene scale divisor relative to window (1=full, 2=half, 4=quarter).",
    )
    args = parser.parse_args(argv)

    pygame.init()
    state.render_w = max(1, config.WIDTH // args.render_scale)
    state.render_h = max(1, config.HEIGHT // args.render_scale)
    state.view_radius = int(config.VIEW_RADIUS)

    if args.renderer == "gl":
        pygame.display.set_mode((config.WIDTH, config.HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        from .gl_draw import begin_lowres_pass, blit_lowres_to_screen, init_lowres_target, setup_ortho
        from .render_gl import draw_frame

        init_lowres_target(state.render_w, state.render_h)
        setup_ortho(state.render_w, state.render_h)
        render_surf = None
    else:
        screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        render_surf = pygame.Surface((state.render_w, state.render_h))
        from .render_soft import draw_frame

    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    while True:
        dt = clock.get_time() / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_KP_PLUS, pygame.K_EQUALS):
                state.view_radius = min(16, state.view_radius + 1)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_KP_MINUS, pygame.K_MINUS):
                state.view_radius = max(1, state.view_radius - 1)

        update_player(dt)

        pcx = int(math.floor(state.cam_pos[0] / config.CHUNK_SIZE))
        pcz = int(math.floor(state.cam_pos[2] / config.CHUNK_SIZE))
        ensure_chunks_around(pcx, pcz)

        update_entities(dt)
        fps = clock.get_fps()

        if args.renderer == "gl":
            begin_lowres_pass(state.render_w, state.render_h)
            draw_frame(pcx, pcz)
            blit_lowres_to_screen(config.WIDTH, config.HEIGHT)
            # Draw HUD at native window resolution so text stays crisp.
            setup_ortho(config.WIDTH, config.HEIGHT)
            draw_status(GLPrimitives(), fps, state.view_radius)
            pygame.display.flip()
        else:
            draw_frame(render_surf, pcx, pcz)
            scaled = pygame.transform.scale(render_surf, (config.WIDTH, config.HEIGHT))
            screen.blit(scaled, (0, 0))
            draw_status(SoftPrimitives(screen), fps, state.view_radius)
            pygame.display.flip()

        clock.tick(config.FPS_LIMIT)


if __name__ == "__main__":
    main()
