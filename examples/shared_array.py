# pyright: strict
"""Reverse each block of an array through `block.SharedArray` scratch memory.

Block-shared memory and `block.sync()` are SIMT/GPU features, so this example
runs on `qd.gpu` (any available GPU backend) rather than the CPU the other
examples use.
"""

import quadrants as qd

import quadrants_typing as qdt

# Threads per block; the reverse happens within each block-sized window.
BLOCK = 4


# Stage each element into shared memory, then read it back mirrored so every
# block-sized window comes out reversed.
@qd.kernel
def block_reverse(
    src: qdt.NDArray[qd.f32, qdt.Dim1],
    dst: qdt.NDArray[qd.f32, qdt.Dim1],
) -> None:
    for i in range(src.shape[0]):
        qd.loop_config(block_dim=BLOCK)
        # `smem` is a `SharedArray[float]`; indexing it returns `f32` (`float`).
        smem = qd.simt.block.SharedArray((BLOCK,), qd.f32)
        t = qd.simt.block.thread_idx()
        smem[t] = src[i]
        qd.simt.block.sync()
        dst[i] = smem[BLOCK - 1 - t]


def main() -> None:
    # Pick whichever GPU backend is available (Metal, CUDA, Vulkan, ...).
    qd.init(arch=qd.gpu)

    # Fill the source with 0..7 so the per-block reversal is easy to read.
    n = 8
    src = qd.ndarray(qd.f32, (n,))
    for i in range(n):
        src[i] = float(i)
    dst = qd.ndarray(qd.f32, (n,))

    block_reverse(src, dst)

    # Each BLOCK-sized window is reversed: [0 1 2 3 | 4 5 6 7] -> [3 2 1 0 | 7 6 5 4].
    print("src:", src.to_numpy())
    print("dst:", dst.to_numpy())


if __name__ == "__main__":
    main()
