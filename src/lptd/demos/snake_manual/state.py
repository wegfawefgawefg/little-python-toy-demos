from __future__ import annotations

from collections import namedtuple

State = namedtuple("State", ["snake", "direction", "food", "score", "game_over"])
# snake: list[(x, y)], head is first element.
# direction: (dx, dy)
# food: (x, y)
# score: int
# game_over: bool


def add_vectors(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


class Functor:
    """Tiny helper for chaining state transforms."""

    def __init__(self, value):
        self.value = value

    def map(self, func):
        return Functor(func(self.value))

    def get(self):
        return self.value

