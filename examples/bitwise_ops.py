# pyright: strict
"""Bitwise operators on integer scalars and integer vectors."""

import quadrants as qd

import quadrants_typing as qdt


@qd.kernel
def scalar_ops(out: qdt.NDArray[qd.u32, qdt.Dim1]) -> None:
    a = qd.u32(12)
    b = qd.u32(10)
    out[0] = a & b
    out[1] = a | b
    out[2] = a ^ b
    out[3] = ~a
    out[4] = a << qd.u32(2)
    out[5] = a >> qd.u32(1)


@qd.kernel
def vector_ops(out: qdt.NDArray[qd.i32, qdt.Dim1]) -> None:
    x = qdt.Vec3i(12, 10, 6)
    y = qdt.Vec3i(10, 3, 5)
    out[0] = (x & y).x
    out[1] = (x | y).y
    out[2] = (x ^ y).z
    out[3] = (~x).x


def main() -> None:
    qd.init(arch=qd.cpu)

    unsigned = qd.ndarray(qd.u32, (6,))
    scalar_ops(unsigned)
    print("scalar a=12 b=10 ->", unsigned.to_numpy())

    signed = qd.ndarray(qd.i32, (4,))
    vector_ops(signed)
    print("vector           ->", signed.to_numpy())


if __name__ == "__main__":
    main()
