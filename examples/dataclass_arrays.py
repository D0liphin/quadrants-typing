# pyright: strict
"""Hold two parallel ndarrays in a dataclass and pass their fields to a kernel."""

from dataclasses import dataclass

import quadrants as qd

import quadrants_typing as qdt


# A pair of same-length arrays: a source and a destination.
@dataclass
class ParallelArrays:
    src: qdt.NDArray[qd.f32, qdt.Dim1]
    dst: qdt.NDArray[qd.f32, qdt.Dim1]


@qd.func
def parallel_arrays_len(pa: ParallelArrays) -> int:
    return pa.src.shape[0]


# Read each source value, double it, and write it to the destination.
@qd.kernel
def double_into(arrays: ParallelArrays) -> None:
    for i in range(parallel_arrays_len(arrays)):
        arrays.dst[i] = arrays.src[i] * 2.0


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Build a source array with ascending values and an empty destination.
    src = qd.ndarray(qd.f32, (6,))
    dst = qd.ndarray(qd.f32, (6,))
    for i in range(6):
        src[i] = float(i)
    arrays = ParallelArrays(src=src, dst=dst)

    # Pass the dataclass fields into the kernel.
    double_into(arrays)

    print("src:", arrays.src.to_numpy())
    print("dst:", arrays.dst.to_numpy())


if __name__ == "__main__":
    main()
