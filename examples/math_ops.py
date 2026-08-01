# pyright: strict
"""Apply a `qd.math` function to every element of an ndarray."""

import quadrants as qd

import quadrants_typing as qdt


# Replace each value with its square root.
@qd.kernel
def sqrt_all[T: (qd.f32, qdt.Vec[qd.f32, qdt.DimAny])](a: qdt.NDArray[T, qdt.Dim1]) -> None:
    for i in range(a.shape[0]):
        a[i] = qd.math.sqrt(a[i])


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Fill an array with perfect squares.
    a = qd.ndarray(qd.f32, (5,))
    for i in range(5):
        a[i] = float((i + 1) ** 2)

    print("before:", a.to_numpy())
    sqrt_all(a)
    print("after: ", a.to_numpy())


if __name__ == "__main__":
    main()
