from __future__ import annotations

import math

import pygame

from . import config, state
from .physics import move_axis_tile


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

    horizontal_forward = (math.sin(state.cam_yaw), 0.0, math.cos(state.cam_yaw))
    right_vec = (horizontal_forward[2], 0.0, -horizontal_forward[0])
    move_vector = (
        forward_input * horizontal_forward[0] + right_input * right_vec[0],
        0.0,
        forward_input * horizontal_forward[2] + right_input * right_vec[2],
    )
    mag = math.sqrt(move_vector[0] ** 2 + move_vector[2] ** 2)
    if mag > 0:
        move_vector = (move_vector[0] / mag, 0.0, move_vector[2] / mag)

    dx = move_vector[0] * config.SPEED * dt
    dz = move_vector[2] * config.SPEED * dt
    state.cam_pos[0] = move_axis_tile(state.cam_pos, dx, 0)
    state.cam_pos[2] = move_axis_tile(state.cam_pos, dz, 2)

    if keys[pygame.K_SPACE] and state.grounded:
        state.v_y = config.JUMP_IMPULSE
        state.grounded = False
    state.v_y += config.GRAVITY * dt
    dy = state.v_y * dt
    orig_y = state.cam_pos[1]
    new_y = move_axis_tile(state.cam_pos, dy, 1)
    if new_y == orig_y and dy < 0:
        state.v_y = 0.0
        state.grounded = True
    else:
        state.grounded = False

