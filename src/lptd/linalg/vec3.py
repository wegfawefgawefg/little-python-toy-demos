import math
import random


class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    @classmethod
    def splat(self, v):
        return Vec3(v, v, v)

    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self):
        mag = self.mag()
        if mag > 0:
            return Vec3(
                self.x / mag,
                self.y / mag,
                self.z / mag,
            )
        return self

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vec3(self.x / other, self.y / other, self.z / other)

    def dot(self, vec3):
        return self.x * vec3.x + self.y * vec3.y + self.z * vec3.z

    def cross(self, vec3):
        return Vec3(
            self.y * vec3.z - self.z * vec3.y,
            self.z * vec3.x - self.x * vec3.z,
            self.x * vec3.y - self.y * vec3.x,
        )

    def __repr__(self):
        return (
            self.x,
            self.y,
            self.z,
        ).__repr__()

    def clone(self):
        return Vec3(
            self.x,
            self.y,
            self.z,
        )

    def clamp(self, low, high):
        return Vec3(
            min(max(self.x, low), high),
            min(max(self.y, low), high),
            min(max(self.z, low), high),
        )

    def rotate_x(self, angle):
        """Rotate around +X by `angle` radians.

        Right-hand rule: with your right thumb pointing +X, positive angles
        rotate +Y toward +Z.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec3(
            self.x,
            self.y * cos_a - self.z * sin_a,
            self.y * sin_a + self.z * cos_a,
        )

    def rotate_y(self, angle):
        """Rotate around +Y by `angle` radians.

        Right-hand rule: with your right thumb pointing +Y, positive angles
        rotate +Z toward +X.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec3(
            self.x * cos_a + self.z * sin_a,
            self.y,
            -self.x * sin_a + self.z * cos_a,
        )

    def rotate_z(self, angle):
        """Rotate around +Z by `angle` radians (same plane rotation as Vec2.rotate).

        Right-hand rule: with your right thumb pointing +Z, positive angles
        rotate +X toward +Y.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec3(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
            self.z,
        )

    def rotate(self, axis, angle):
        """Rotate around `axis` by `angle` radians (axis-angle).

        Uses Rodrigues' rotation formula. `axis` can be any (non-zero) Vec3 and
        will be normalized internally.
        """
        k = axis.norm()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return (
            self * cos_a
            + (k.cross(self) * sin_a)
            + (k * (k.dot(self) * (1 - cos_a)))
        )

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_vec4(self, w=1.0):
        # Local import to avoid cycles (Vec4 imports Vec3).
        from lptd.linalg.vec4 import Vec4

        return Vec4(self.x, self.y, self.z, w)

    @classmethod
    def random(self):
        return Vec3(
            random.random(),
            random.random(),
            random.random(),
        )
