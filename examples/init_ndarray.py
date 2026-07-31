# pyright: strict
"""Allocate an empty ndarray and fill it from a kernel."""

import quadrants as qd

import quadrants_typing as qdt


# Write a value into every slot of a 1-D `f32` array.
@qd.kernel
def fill(a: qdt.NDArray[qd.f32, qdt.Dim1]) -> None:
    for i in range(a.shape[0]):
        a[i] = i * 2.0


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Reserve storage for eight `f32` values.
    a = qd.ndarray(qd.f32, (8,))

    # Launch the kernel to populate the array.
    fill(a)

    # Copy the array back to the host and print it.
    print(a.to_numpy())


if __name__ == "__main__":
    main()
