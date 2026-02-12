import math

from lptd.linalg.vec3 import Vec3


class Mat3:
    """3x3 matrix (row-major).

    This is the "standard" 3D 3x3 used for rotation/scale and normal matrices.
    Vectors are treated as column vectors:
        v' = M @ v
    """

    def __init__(self, m=None):
        # Default to identity.
        if m is None:
            self.m = [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        else:
            if len(m) != 9:
                raise ValueError("Mat3 expects 9 elements")
            self.m = [float(x) for x in m]

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def scale(cls, sx, sy=None, sz=None):
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        return cls(
            [
                float(sx),
                0.0,
                0.0,
                0.0,
                float(sy),
                0.0,
                0.0,
                0.0,
                float(sz),
            ]
        )

    @classmethod
    def rotate_x(cls, angle):
        """Rotate around +X by `angle` radians (right-hand rule)."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                c,
                -s,
                0.0,
                s,
                c,
            ]
        )

    @classmethod
    def rotate_y(cls, angle):
        """Rotate around +Y by `angle` radians (right-hand rule)."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                c,
                0.0,
                s,
                0.0,
                1.0,
                0.0,
                -s,
                0.0,
                c,
            ]
        )

    @classmethod
    def rotate_z(cls, angle):
        """Rotate around +Z by `angle` radians (right-hand rule)."""
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                c,
                -s,
                0.0,
                s,
                c,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def rotate(cls, axis, angle):
        """Axis-angle rotation (Rodrigues). Axis is normalized internally."""
        k = axis.norm()
        x, y, z = float(k.x), float(k.y), float(k.z)
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        return cls(
            [
                t * x * x + c,
                t * x * y - s * z,
                t * x * z + s * y,
                t * x * y + s * z,
                t * y * y + c,
                t * y * z - s * x,
                t * x * z - s * y,
                t * y * z + s * x,
                t * z * z + c,
            ]
        )

    def clone(self):
        return Mat3(self.m)

    def __repr__(self):
        r0 = self.m[0:3]
        r1 = self.m[3:6]
        r2 = self.m[6:9]
        return f"Mat3({r0}, {r1}, {r2})"

    def to_tuple(self):
        return (
            (self.m[0], self.m[1], self.m[2]),
            (self.m[3], self.m[4], self.m[5]),
            (self.m[6], self.m[7], self.m[8]),
        )

    def transpose(self):
        m = self.m
        return Mat3(
            [
                m[0],
                m[3],
                m[6],
                m[1],
                m[4],
                m[7],
                m[2],
                m[5],
                m[8],
            ]
        )

    def det(self):
        m = self.m
        return (
            m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
        )

    def inverse(self):
        m = self.m
        d = self.det()
        if d == 0.0:
            raise ValueError("Mat3 is singular")
        invd = 1.0 / d
        return Mat3(
            [
                (m[4] * m[8] - m[5] * m[7]) * invd,
                (m[2] * m[7] - m[1] * m[8]) * invd,
                (m[1] * m[5] - m[2] * m[4]) * invd,
                (m[5] * m[6] - m[3] * m[8]) * invd,
                (m[0] * m[8] - m[2] * m[6]) * invd,
                (m[2] * m[3] - m[0] * m[5]) * invd,
                (m[3] * m[7] - m[4] * m[6]) * invd,
                (m[1] * m[6] - m[0] * m[7]) * invd,
                (m[0] * m[4] - m[1] * m[3]) * invd,
            ]
        )

    def _mul_mat3(self, other):
        a = self.m
        b = other.m
        out = [0.0] * 9
        for r in range(3):
            for c in range(3):
                out[r * 3 + c] = (
                    a[r * 3 + 0] * b[0 * 3 + c]
                    + a[r * 3 + 1] * b[1 * 3 + c]
                    + a[r * 3 + 2] * b[2 * 3 + c]
                )
        return Mat3(out)

    def transform(self, v):
        x = float(v.x)
        y = float(v.y)
        z = float(v.z)
        m = self.m
        return Vec3(
            m[0] * x + m[1] * y + m[2] * z,
            m[3] * x + m[4] * y + m[5] * z,
            m[6] * x + m[7] * y + m[8] * z,
        )

    def __matmul__(self, other):
        if isinstance(other, Mat3):
            return self._mul_mat3(other)
        if isinstance(other, Vec3):
            return self.transform(other)
        raise TypeError(f"unsupported operand type(s) for @: 'Mat3' and '{type(other)}'")
