# pyright: strict
"""Define `Point` and `Affine` types and transform points with a `@` matrix multiply."""

import quadrants as qd

import quadrants_typing as qdt

# A 3-D point stored as three `f32` components.
Point = qdt.Vec[qd.f32, qdt.Dim3]

# A 3x4 affine transform: a 3x3 linear part plus a translation column.
Affine = qdt.Mat[qd.f32, qdt.Dim3, qdt.Dim4]


# Copy the 3x4 affine into a 4x4 matrix and append the row `[0, 0, 0, 1]`.
@qd.func
def to_mat4(a: Affine) -> qdt.Mat4f:
    return qdt.Mat4f(
        [
            [a[0, 0], a[0, 1], a[0, 2], a[0, 3]],
            [a[1, 0], a[1, 1], a[1, 2], a[1, 3]],
            [a[2, 0], a[2, 1], a[2, 2], a[2, 3]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# Transform every point by the affine using a single 4x4 matrix multiply each.
@qd.kernel
def transform(points: qdt.NDArray[Point, qdt.Dim1], a: Affine) -> None:
    m = to_mat4(a)
    for i in range(points.shape[0]):
        p = points[i]
        # Lift the point to homogeneous coordinates so `@` applies translation too.
        homogeneous = qd.math.vec4(p.x, p.y, p.z, 1.0)
        result = m @ homogeneous
        points[i] = qd.math.vec3(result.x, result.y, result.z)


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Place three points along the diagonal.
    points = qd.ndarray(Point, (3,))
    for i in range(3):
        points[i] = qd.math.vec3(i)

    # Scale every axis by 2 and translate by (1, 2, 3).
    affine = Affine(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
        ]
    )

    print("before:\n", points.to_numpy())
    transform(points, affine)
    print("after:\n", points.to_numpy())


if __name__ == "__main__":
    main()
