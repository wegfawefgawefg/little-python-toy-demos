from __future__ import annotations

import random

import pygame

from . import config
from .state import Functor, State, add_vectors


def handle_input(state: State, events) -> State:
    key_map = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
    }
    for event in events:
        if event.type == pygame.KEYDOWN:
            new_dir = key_map.get(event.key)
            if new_dir and add_vectors(state.direction, new_dir) != (0, 0):
                state = state._replace(direction=new_dir)
    return state


def move_snake(state: State) -> State:
    new_head = add_vectors(state.snake[0], state.direction)
    if new_head == state.food:
        new_snake = [new_head] + state.snake
    else:
        new_snake = [new_head] + state.snake[:-1]
    return state._replace(snake=new_snake)


def check_collisions(state: State) -> State:
    x, y = state.snake[0]
    if x < 0 or x >= config.GRID_WIDTH or y < 0 or y >= config.GRID_HEIGHT:
        return state._replace(game_over=True)
    if len(state.snake) > 1 and state.snake[0] in state.snake[1:]:
        return state._replace(game_over=True)
    return state


def update_food_and_score(state: State) -> State:
    if state.snake[0] != state.food:
        return state
    new_food = (
        random.randint(0, config.GRID_WIDTH - 1),
        random.randint(0, config.GRID_HEIGHT - 1),
    )
    return state._replace(food=new_food, score=state.score + 1)


def game_tick(state: State, events) -> State:
    return (
        Functor(state)
        .map(lambda s: handle_input(s, events))
        .map(move_snake)
        .map(check_collisions)
        .map(lambda s: update_food_and_score(s) if not s.game_over else s)
        .get()
    )

