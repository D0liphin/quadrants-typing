# pyright: strict
"""Give plain scalars a checker-only `Literal` type with `qdt.annotations.Runtime`."""

from typing import Literal

import quadrants as qd

import quadrants_typing as qdt

# `Runtime[TypeChecker, runtime]` reads as `TypeChecker` to the type checker but
# collapses to `runtime` when quadrants inspects the annotation. Here the checker
# sees a restricted enum-like `Literal` set, while the runtime dtype is `int`.
#
# Use a plain assignment (not `type Channel = ...`): a PEP 695 alias stays lazy
# at runtime, but a plain alias resolves eagerly, so quadrants sees `int`.
BlockChannel = qdt.annotations.Runtime[Literal[0, 1, 2, 3, 4, 5, 6], int]

# Enum-style constants. The type checker rejects any value outside the literal
# set (e.g. `BLOCK_BAD: BlockChannel = 7` would error), yet each is a plain int.
BLOCK_TIMESTEPS: BlockChannel = 0
BLOCK_GX: BlockChannel = 3
BLOCK_ADC: BlockChannel = 6


# `BlockChannel` works as a scalar kernel-argument annotation: the checker sees
# the `Literal` set, quadrants sees the resolved `int` dtype.
@qd.kernel
def next_channel(ch: BlockChannel) -> qd.i32:
    return ch + 1


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    print("timesteps -> next:", next_channel(BLOCK_TIMESTEPS))
    print("gx -> next:", next_channel(BLOCK_GX))
    print("adc -> next:", next_channel(BLOCK_ADC))


if __name__ == "__main__":
    main()
