# pyright: strict
"""Use `Template[bool]` with `qd.static` to compile one branch of a loop body."""

import quadrants as qd

import quadrants_typing as qdt


# `double` is known at compile time, so `qd.static` picks which line compiles in.
@qd.kernel
def scale(a: qdt.NDArray[qd.f32, qdt.Dim1], double: qdt.Template[bool]) -> None:
    for i in range(a.shape[0]):
        if qd.static(double):
            a[i] = a[i] * 2.0
        else:
            a[i] = a[i] + 1.0


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Fill an array with threes for each specialization to act on.
    doubled = qd.ndarray(qd.f32, (5,))
    incremented = qd.ndarray(qd.f32, (5,))
    doubled.fill(3.0)
    incremented.fill(3.0)

    # Passing `True` compiles the doubling branch.
    scale(doubled, True)

    # Passing `False` compiles the increment branch as a separate specialization.
    scale(incremented, False)

    print("double=True :", doubled.to_numpy())
    print("double=False:", incremented.to_numpy())


if __name__ == "__main__":
    main()
