import math

from lptd.linalg.vec3 import Vec3
from lptd.linalg.vec4 import Vec4
from lptd.linalg.mat3 import Mat3


class Mat4:
    """4x4 matrix (row-major).

    Vectors are treated as column vectors:
        p' = M @ (x, y, z, w)

    Helpers:
    - transform_point(v): uses w=1, divides by w
    - transform_vector(v): uses w=0 (no translation)
    """

    def __init__(self, m=None):
        if m is None:
            self.m = [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        else:
            if len(m) != 16:
                raise ValueError("Mat4 expects 16 elements")
            self.m = [float(x) for x in m]

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def from_mat3(cls, m3, translation=None):
        """Embed a Mat3 into the upper-left of a Mat4.

        If `translation` is provided (Vec3), it is placed in the last column.
        """
        t = translation if translation is not None else Vec3(0.0, 0.0, 0.0)
        m = m3.m
        return cls(
            [
                m[0],
                m[1],
                m[2],
                float(t.x),
                m[3],
                m[4],
                m[5],
                float(t.y),
                m[6],
                m[7],
                m[8],
                float(t.z),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def translate(cls, tx, ty, tz):
        return cls(
            [
                1.0,
                0.0,
                0.0,
                float(tx),
                0.0,
                1.0,
                0.0,
                float(ty),
                0.0,
                0.0,
                1.0,
                float(tz),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

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
                0.0,
                float(sy),
                0.0,
                0.0,
                0.0,
                0.0,
                float(sz),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def rotate_x(cls, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                c,
                -s,
                0.0,
                0.0,
                s,
                c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def rotate_y(cls, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                c,
                0.0,
                s,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -s,
                0.0,
                c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def rotate_z(cls, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return cls(
            [
                c,
                -s,
                0.0,
                0.0,
                s,
                c,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def rotate(cls, axis, angle):
        r3 = Mat3.rotate(axis, angle)
        m = r3.m
        return cls(
            [
                m[0],
                m[1],
                m[2],
                0.0,
                m[3],
                m[4],
                m[5],
                0.0,
                m[6],
                m[7],
                m[8],
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def perspective(cls, fov_y, aspect, near, far):
        """Right-handed perspective matrix.

        fov_y in radians. near/far > 0.
        Maps to OpenGL-style clip space: z in [-w, +w] after projection.
        """
        f = 1.0 / math.tan(fov_y * 0.5)
        n = float(near)
        fa = float(far)
        if aspect == 0:
            raise ValueError("aspect must be non-zero")
        if n <= 0 or fa <= 0 or n == fa:
            raise ValueError("invalid near/far")
        return cls(
            [
                f / float(aspect),
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                (fa + n) / (n - fa),
                (2.0 * fa * n) / (n - fa),
                0.0,
                0.0,
                -1.0,
                0.0,
            ]
        )

    @classmethod
    def ortho(cls, left, right, bottom, top, near, far):
        l = float(left)
        r = float(right)
        b = float(bottom)
        t = float(top)
        n = float(near)
        f = float(far)
        if l == r or b == t or n == f:
            raise ValueError("invalid ortho volume")
        return cls(
            [
                2.0 / (r - l),
                0.0,
                0.0,
                -(r + l) / (r - l),
                0.0,
                2.0 / (t - b),
                0.0,
                -(t + b) / (t - b),
                0.0,
                0.0,
                -2.0 / (f - n),
                -(f + n) / (f - n),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    @classmethod
    def look_at(cls, eye, target, up):
        """Right-handed look-at view matrix."""
        f = (target - eye).norm()
        s = f.cross(up).norm()
        u = s.cross(f)

        ex = -s.dot(eye)
        ey = -u.dot(eye)
        ez = f.dot(eye)

        return cls(
            [
                s.x,
                s.y,
                s.z,
                ex,
                u.x,
                u.y,
                u.z,
                ey,
                -f.x,
                -f.y,
                -f.z,
                ez,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

    def clone(self):
        return Mat4(self.m)

    def __repr__(self):
        m = self.m
        return f"Mat4({m[0:4]}, {m[4:8]}, {m[8:12]}, {m[12:16]})"

    def to_tuple(self):
        m = self.m
        return (
            (m[0], m[1], m[2], m[3]),
            (m[4], m[5], m[6], m[7]),
            (m[8], m[9], m[10], m[11]),
            (m[12], m[13], m[14], m[15]),
        )

    def transpose(self):
        m = self.m
        return Mat4(
            [
                m[0],
                m[4],
                m[8],
                m[12],
                m[1],
                m[5],
                m[9],
                m[13],
                m[2],
                m[6],
                m[10],
                m[14],
                m[3],
                m[7],
                m[11],
                m[15],
            ]
        )

    def _mul_mat4(self, other):
        a = self.m
        b = other.m
        out = [0.0] * 16
        for r in range(4):
            for c in range(4):
                out[r * 4 + c] = (
                    a[r * 4 + 0] * b[0 * 4 + c]
                    + a[r * 4 + 1] * b[1 * 4 + c]
                    + a[r * 4 + 2] * b[2 * 4 + c]
                    + a[r * 4 + 3] * b[3 * 4 + c]
                )
        return Mat4(out)

    def transform_point(self, v):
        x = float(v.x)
        y = float(v.y)
        z = float(v.z)
        m = self.m
        nx = m[0] * x + m[1] * y + m[2] * z + m[3]
        ny = m[4] * x + m[5] * y + m[6] * z + m[7]
        nz = m[8] * x + m[9] * y + m[10] * z + m[11]
        nw = m[12] * x + m[13] * y + m[14] * z + m[15]
        if nw != 0.0:
            invw = 1.0 / nw
            nx *= invw
            ny *= invw
            nz *= invw
        return Vec3(nx, ny, nz)

    def transform_vector(self, v):
        x = float(v.x)
        y = float(v.y)
        z = float(v.z)
        m = self.m
        return Vec3(
            m[0] * x + m[1] * y + m[2] * z,
            m[4] * x + m[5] * y + m[6] * z,
            m[8] * x + m[9] * y + m[10] * z,
        )

    def transform_vec4(self, v):
        x = float(v.x)
        y = float(v.y)
        z = float(v.z)
        w = float(v.w)
        m = self.m
        return Vec4(
            m[0] * x + m[1] * y + m[2] * z + m[3] * w,
            m[4] * x + m[5] * y + m[6] * z + m[7] * w,
            m[8] * x + m[9] * y + m[10] * z + m[11] * w,
            m[12] * x + m[13] * y + m[14] * z + m[15] * w,
        )

    def upper_left_mat3(self):
        m = self.m
        return Mat3(
            [
                m[0],
                m[1],
                m[2],
                m[4],
                m[5],
                m[6],
                m[8],
                m[9],
                m[10],
            ]
        )

    def normal_matrix(self):
        """Normal matrix = inverse(transpose(upper-left 3x3))."""
        return self.upper_left_mat3().inverse().transpose()

    def __matmul__(self, other):
        if isinstance(other, Mat4):
            return self._mul_mat4(other)
        if isinstance(other, Vec3):
            return self.transform_point(other)
        if isinstance(other, Vec4):
            return self.transform_vec4(other)
        raise TypeError(
            f"unsupported operand type(s) for @: 'Mat4' and '{type(other)}'"
        )
