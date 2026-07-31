# pyright: strict
"""Build a histogram in parallel with `qd.atomic_add`."""

import quadrants as qd

import quadrants_typing as qdt


# Count how many samples fall into each bin, adding safely across threads.
@qd.kernel
def histogram(
    samples: qdt.NDArray[qd.i32, qdt.Dim1],
    bins: qdt.NDArray[qd.i32, qdt.Dim1],
) -> None:
    for i in range(samples.shape[0]):
        bin_index = samples[i]
        qd.atomic_add(bins[bin_index], 1)


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Assign each sample a bin in the range 0..3.
    n = 12
    samples = qd.ndarray(qd.i32, (n,))
    for i in range(n):
        samples[i] = i % 4

    # Start every bin count at zero.
    bins = qd.ndarray(qd.i32, (4,))
    bins.fill(0)

    histogram(samples, bins)

    print("samples:", samples.to_numpy())
    print("bins:   ", bins.to_numpy())


if __name__ == "__main__":
    main()
