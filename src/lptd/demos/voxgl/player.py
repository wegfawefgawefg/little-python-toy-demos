from __future__ import annotations

import math

import pygame

from . import config, state


def update(dt: float) -> None:
    mx, my = pygame.mouse.get_rel()
    state.cam_yaw += mx * config.MOUSE_SENSITIVITY
    state.cam_pitch -= my * config.MOUSE_SENSITIVITY
    state.cam_pitch = max(
        -math.pi / 2 + 0.01,
        min(math.pi / 2 - 0.01, state.cam_pitch),
    )

    keys = pygame.key.get_pressed()
    forward_input = 0.0
    right_input = 0.0
    if keys[pygame.K_w]:
        forward_input += 1.0
    if keys[pygame.K_s]:
        forward_input -= 1.0
    if keys[pygame.K_d]:
        right_input += 1.0
    if keys[pygame.K_a]:
        right_input -= 1.0

    vertical_input = 0.0
    if keys[pygame.K_SPACE]:
        vertical_input += 1.0
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        vertical_input -= 1.0

    horizontal_forward = (math.sin(state.cam_yaw), 0.0, math.cos(state.cam_yaw))
    right_vec = (horizontal_forward[2], 0.0, -horizontal_forward[0])
    move_vector = (
        forward_input * horizontal_forward[0] + right_input * right_vec[0],
        vertical_input,
        forward_input * horizontal_forward[2] + right_input * right_vec[2],
    )
    mag = math.sqrt(move_vector[0] ** 2 + move_vector[1] ** 2 + move_vector[2] ** 2)
    if mag > 0:
        move_vector = (move_vector[0] / mag, move_vector[1] / mag, move_vector[2] / mag)

    speed = config.SPEED
    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
        speed *= config.FAST_MOVE_MULT

    dx = move_vector[0] * speed * dt
    dy = move_vector[1] * speed * dt
    dz = move_vector[2] * speed * dt
    state.cam_pos[0] += dx
    state.cam_pos[1] += dy
    state.cam_pos[2] += dz
