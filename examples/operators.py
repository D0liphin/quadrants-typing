# pyright: strict
"""A taste of the operators the typing layer declares, across scalars, vectors and matrices."""

import quadrants as qd

import quadrants_typing as qdt


@qd.kernel
def scalar_operators(out: qdt.NDArray[qd.i32, qdt.Dim1]) -> None:
    a = qd.i32(12)
    b = qd.i32(10)
    out[0] = a + b
    out[1] = a < b
    out[2] = a & b


@qd.kernel
def shift_operators(out: qdt.NDArray[qd.u32, qdt.Dim1], n: qd.i32) -> None:
    out[0] = qd.cast(qd.u64(n) >> 32, qd.u32)
    out[1] = qd.u32(n) >> 8
    out[2] = qd.u32(1) << n
    out[3] = qd.cast((n >> 31) & 1, qd.u32)
    out[4] = (qd.math.uvec3(1, 2, 3) << n).z


@qd.kernel
def vector_operators(out: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    x = qdt.Vec3f(6.0, 4.0, 2.0)
    y = qdt.Vec3f(3.0, 2.0, 1.0)
    s = qd.f32(2.0)
    out[0] = (x + y).x
    out[1] = (x * s).y
    out[2] = qd.cast((x < y).z, qd.f32)


@qd.kernel
def matrix_operators(out: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    m = qdt.Mat3f([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    v = qdt.Vec3f(1.0, 2.0, 3.0)
    s = qd.f32(2.0)
    out[0] = (m * s)[2, 2]
    out[1] = (m @ v).z


def main() -> None:
    qd.init(arch=qd.cpu)

    scalars = qd.ndarray(qd.i32, (3,))
    scalar_operators(scalars)
    print("scalar (a=12, b=10):", scalars.to_numpy())

    shifts = qd.ndarray(qd.u32, (5,))
    shift_operators(shifts, 4)

    vectors = qd.ndarray(qd.f32, (3,))
    vector_operators(vectors)
    print("vector (x=(6,4,2), y=(3,2,1), s=2):", vectors.to_numpy())

    matrices = qd.ndarray(qd.f32, (2,))
    matrix_operators(matrices)
    print("matrix (m=diag(1,2,3), v=(1,2,3), s=2):", matrices.to_numpy())


if __name__ == "__main__":
    main()
