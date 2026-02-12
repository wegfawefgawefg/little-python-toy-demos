import math
import random

from lptd.linalg.vec3 import Vec3


class Vec4:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x, self.y, self.z, self.w = x, y, z, w

    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def norm(self):
        mag = self.mag()
        if mag > 0:
            return Vec4(
                self.x / mag,
                self.y / mag,
                self.z / mag,
                self.w / mag,
            )
        return self

    def __add__(self, other):
        return Vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other):
        return Vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, other):
        if isinstance(other, Vec4):
            return Vec4(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w)
        return Vec4(self.x * other, self.y * other, self.z * other, self.w * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Vec4):
            return Vec4(self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w)
        return Vec4(self.x / other, self.y / other, self.z / other, self.w / other)

    def dot(self, vec4):
        return self.x * vec4.x + self.y * vec4.y + self.z * vec4.z + self.w * vec4.w

    def __repr__(self):
        return (
            self.x,
            self.y,
            self.z,
            self.w,
        ).__repr__()

    def clone(self):
        return Vec4(
            self.x,
            self.y,
            self.z,
            self.w,
        )

    def clamp(self, low, high):
        return Vec4(
            min(max(self.x, low), high),
            min(max(self.y, low), high),
            min(max(self.z, low), high),
            min(max(self.w, low), high),
        )

    def to_tuple(self):
        return (self.x, self.y, self.z, self.w)

    def xyz(self):
        return Vec3(self.x, self.y, self.z)

    def to_vec3(self, perspective_divide=True):
        if perspective_divide and self.w != 0:
            invw = 1.0 / self.w
            return Vec3(self.x * invw, self.y * invw, self.z * invw)
        return Vec3(self.x, self.y, self.z)

    @classmethod
    def random(self):
        return Vec4(
            random.random(),
            random.random(),
            random.random(),
            random.random(),
        )
