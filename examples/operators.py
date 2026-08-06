# pyright: strict
"""Every operator the typing layer declares, on scalars, vectors and matrices."""

from typing import Any

import numpy.typing as npt
import quadrants as qd

import quadrants_typing as qdt

SCALAR_ARITHMETIC = (
    "a + b",
    "a - b",
    "a * b",
    "a / b",
    "a // b",
    "a % b",
    "a**b",
    "-a",
    "+a",
    "abs(-a)",
)
SCALAR_COMPARISON = ("a < b", "a <= b", "a > b", "a >= b")
SCALAR_BITWISE = ("a & b", "a | b", "a ^ b", "~a", "a << 2", "a >> 1")
VECTOR_ARITHMETIC = (
    "(x + y).x",
    "(x - y).y",
    "(x * y).z",
    "(x / y).x",
    "(x // y).x",
    "(x % y).x",
    "(x**y).x",
    "(-x).x",
    "(+x).x",
    "abs(-x).x",
    "(x + s).x",
    "(s + x).x",
    "(x - s).x",
    "(s - x).x",
    "(x * s).y",
    "(s * x).y",
    "(x / s).z",
    "(s / x).z",
    "(x // s).x",
    "(s // x).x",
    "(x % s).x",
    "(s % x).x",
    "(x**s).x",
    "(s**x).x",
    "x[1] = s; x[1]",
)
VECTOR_COMPARISON = (
    "(x < y).x",
    "(x <= y).z",
    "(x > y).x",
    "(x >= y).z",
    "(x == y).x",
    "(x == y).y",
    "(x != y).x",
    "(x != y).y",
)
VECTOR_BITWISE = (
    "(x & y).x",
    "(x | y).y",
    "(x ^ y).z",
    "(~x).x",
    "(x << sh).x",
    "(x >> sh).x",
    "(x & 6).x",
    "(6 & x).x",
    "(x | 1).x",
    "(1 | x).x",
    "(x ^ 5).x",
    "(5 ^ x).x",
    "(x << 2).x",
    "(2 << x).x",
    "(x >> 2).x",
    "(4096 >> x).x",
)
MATRIX = (
    "(m + n)[2, 2]",
    "(m - n)[2, 2]",
    "(-m)[2, 2]",
    "(m * n)[1, 1]",
    "(m * s)[2, 2]",
    "(s * m)[2, 2]",
    "(m / s)[2, 2]",
    "(m @ m)[2, 2]",
    "(m @ v).z",
    "m[0, 0] = s; m[0, 0]",
)


@qd.kernel
def scalar_arithmetic(out: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    a = qd.f32(6.0)
    b = qd.f32(3.0)
    out[0] = a + b
    out[1] = a - b
    out[2] = a * b
    out[3] = a / b
    out[4] = a // b
    out[5] = a % b
    out[6] = a**b
    out[7] = -a
    out[8] = +a
    out[9] = abs(-a)


# A comparison yields a `bool`, which is an `int` — hence the `i32` output.
@qd.kernel
def scalar_comparison(out: qdt.NDArray[qd.i32, qdt.Dim1]) -> None:
    a = qd.f32(6.0)
    b = qd.f32(3.0)
    out[0] = a < b
    out[1] = a <= b
    out[2] = a > b
    out[3] = a >= b


# The bitwise operators are declared on `Integer`; `u32` shows `~` unsigned.
@qd.kernel
def scalar_bitwise(out: qdt.NDArray[qd.u32, qdt.Dim1]) -> None:
    a = qd.u32(12)
    b = qd.u32(10)
    out[0] = a & b
    out[1] = a | b
    out[2] = a ^ b
    out[3] = ~a
    out[4] = a << qd.u32(2)
    out[5] = a >> qd.u32(1)


@qd.kernel
def vector_arithmetic(out: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    x = qdt.Vec3f(6.0, 4.0, 2.0)
    y = qdt.Vec3f(3.0, 2.0, 1.0)
    s = qd.f32(2.0)
    out[0] = (x + y).x
    out[1] = (x - y).y
    out[2] = (x * y).z
    out[3] = (x / y).x
    out[4] = (x // y).x
    out[5] = (x % y).x
    out[6] = (x**y).x
    out[7] = (-x).x
    out[8] = (+x).x
    out[9] = abs(-x).x
    # Elementwise against a scalar, each with its reflected form.
    out[10] = (x + s).x
    out[11] = (s + x).x
    out[12] = (x - s).x
    out[13] = (s - x).x
    out[14] = (x * s).y
    out[15] = (s * x).y
    out[16] = (x / s).z
    out[17] = (s / x).z
    out[18] = (x // s).x
    out[19] = (s // x).x
    out[20] = (x % s).x
    out[21] = (s % x).x
    out[22] = (x**s).x
    out[23] = (s**x).x
    x[1] = s
    out[24] = x[1]


# Every comparison is elementwise, so it yields a `Vec[u1, D]` mask rather than
# the scalar `bool` a comparison of two scalars gives.
@qd.kernel
def vector_comparison(out: qdt.NDArray[qd.u1, qdt.Dim1]) -> None:
    x = qdt.Vec3f(6.0, 4.0, 2.0)
    y = qdt.Vec3f(6.0, 2.0, 4.0)
    out[0] = (x < y).x
    out[1] = (x <= y).z
    out[2] = (x > y).x
    out[3] = (x >= y).z
    out[4] = (x == y).x
    out[5] = (x == y).y
    out[6] = (x != y).x
    out[7] = (x != y).y


@qd.kernel
def vector_bitwise(out: qdt.NDArray[qd.i32, qdt.Dim1]) -> None:
    x = qdt.Vec3i(12, 10, 6)
    y = qdt.Vec3i(10, 3, 5)
    sh = qdt.Vec3i(2, 1, 3)
    out[0] = (x & y).x
    out[1] = (x | y).y
    out[2] = (x ^ y).z
    out[3] = (~x).x
    out[4] = (x << sh).x
    out[5] = (x >> sh).x
    # Against a scalar, each with its reflected form. A reflected shift reads
    # oddly — the vector is the shift count, not the value being shifted.
    out[6] = (x & 6).x
    out[7] = (6 & x).x
    out[8] = (x | 1).x
    out[9] = (1 | x).x
    out[10] = (x ^ 5).x
    out[11] = (5 ^ x).x
    out[12] = (x << 2).x
    out[13] = (2 << x).x
    out[14] = (x >> 2).x
    out[15] = (4096 >> x).x


@qd.kernel
def matrix_operators(out: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    m = qdt.Mat3f([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    # A bare scalar fills every entry.
    n = qdt.Mat3f(1.0)
    v = qdt.Vec3f(1.0, 2.0, 3.0)
    s = qd.f32(2.0)
    out[0] = (m + n)[2, 2]
    out[1] = (m - n)[2, 2]
    out[2] = (-m)[2, 2]
    # `*` is elementwise or a scale; the matrix product is `@`.
    out[3] = (m * n)[1, 1]
    out[4] = (m * s)[2, 2]
    out[5] = (s * m)[2, 2]
    out[6] = (m / s)[2, 2]
    out[7] = (m @ m)[2, 2]
    out[8] = (m @ v).z
    m[0, 0] = s
    out[9] = m[0, 0]


def report(title: str, labels: tuple[str, ...], values: npt.NDArray[Any]) -> None:
    print(f"{title}:")
    for label, value in zip(labels, values.tolist()):
        print(f"  {label:>21} = {value}")


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    scalars = qd.ndarray(qd.f32, (len(SCALAR_ARITHMETIC),))
    scalar_arithmetic(scalars)
    report("scalar arithmetic (a=6, b=3)", SCALAR_ARITHMETIC, scalars.to_numpy())

    flags = qd.ndarray(qd.i32, (len(SCALAR_COMPARISON),))
    scalar_comparison(flags)
    report("scalar comparison (a=6, b=3)", SCALAR_COMPARISON, flags.to_numpy())

    unsigned = qd.ndarray(qd.u32, (len(SCALAR_BITWISE),))
    scalar_bitwise(unsigned)
    report("scalar bitwise, u32 (a=12, b=10)", SCALAR_BITWISE, unsigned.to_numpy())

    vectors = qd.ndarray(qd.f32, (len(VECTOR_ARITHMETIC),))
    vector_arithmetic(vectors)
    report(
        "vector arithmetic (x=(6,4,2), y=(3,2,1), s=2)",
        VECTOR_ARITHMETIC,
        vectors.to_numpy(),
    )

    bits = qd.ndarray(qd.u1, (len(VECTOR_COMPARISON),))
    vector_comparison(bits)
    report("vector comparison (x=(6,4,2), y=(6,2,4))", VECTOR_COMPARISON, bits.to_numpy())

    vector_ints = qd.ndarray(qd.i32, (len(VECTOR_BITWISE),))
    vector_bitwise(vector_ints)
    report(
        "vector bitwise (x=(12,10,6), y=(10,3,5), sh=(2,1,3))",
        VECTOR_BITWISE,
        vector_ints.to_numpy(),
    )

    matrices = qd.ndarray(qd.f32, (len(MATRIX),))
    matrix_operators(matrices)
    report("matrix (m=diag(1,2,3), n=ones, v=(1,2,3), s=2)", MATRIX, matrices.to_numpy())


if __name__ == "__main__":
    main()
