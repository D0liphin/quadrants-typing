# pyright: strict
"""Template one add kernel over element type and array rank, iterating with `qd.grouped`."""

import quadrants as qd

import quadrants_typing as qdt


# The element type is a scalar number or a vector of numbers, and the rank is
# left open. `qd.grouped` yields one index `I` covering every element, of any rank.
@qd.kernel
def add[T: (qdt.Number, int, float), D: qdt.DimAny](
    a: qdt.NDArray[T, D],
    b: qdt.NDArray[T, D],
    out: qdt.NDArray[T, D],
) -> None:
    for index in qd.grouped(out):
        out[index] = a[index] + b[index]


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Add two 1-D `f32` arrays, specializing the kernel for rank 1.
    a1 = qd.ndarray(qd.f32, (4,))
    b1 = qd.ndarray(qd.f32, (4,))
    out1 = qd.ndarray(qd.f32, (4,))
    a1.fill(1.0)
    b1.fill(2.0)
    add(a1, b1, out1)
    print("1-D:", out1.to_numpy())

    # Add two 2-D `f32` arrays, specializing the same kernel for rank 2.
    a2 = qd.ndarray(qd.f32, (2, 3))
    b2 = qd.ndarray(qd.f32, (2, 3))
    out2 = qd.ndarray(qd.f32, (2, 3))
    a2.fill(10.0)
    b2.fill(5.0)
    add(a2, b2, out2)
    print("2-D:", out2.to_numpy())


if __name__ == "__main__":
    main()
