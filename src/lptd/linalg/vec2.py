import math
import random


class Vec2:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y

    @classmethod
    def splat(self, v):
        return Vec2(v, v)

    def mag(self):
        return math.sqrt(self.x**2 + self.y**2)

    def norm(self):
        mag = self.mag()
        if mag > 0:
            return Vec2(
                self.x / mag,
                self.y / mag,
            )
        return self

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        return Vec2(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x / other.x, self.y / other.y)
        return Vec2(self.x / other, self.y / other)

    def dot(self, vec2):
        return self.x * vec2.x + self.y * vec2.y

    def cross(self, vec2):
        return self.x * vec2.y - self.y * vec2.x

    def __repr__(self):
        return (
            self.x,
            self.y,
        ).__repr__()

    def clone(self):
        return Vec2(
            self.x,
            self.y,
        )

    def clamp(self, low, high):
        return Vec2(
            min(max(self.x, low), high),
            min(max(self.y, low), high),
        )

    def rotate(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def to_tuple(self):
        return (self.x, self.y)

    @classmethod
    def random(self):
        return Vec2(
            random.random(),
            random.random(),
        )
