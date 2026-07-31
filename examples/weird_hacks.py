# pyright: strict
"""
You **may** (keyword 'may') want to supply exactly the runtime value you want
to quadrants annotations, here is how you can do this.

There is probably no good use for this. This is a true 'hack' and not very nice.
If you find that you actually need lots of vectors of dimension >8, do post an
issue! we're happy to add them.
"""

import quadrants as qd

import quadrants_typing as qdt

Vec20s = qdt.Vec[qd.i16, qdt.annotations.Runtime[qdt.DimAny, 20]]


@qd.kernel
def identity(v: Vec20s) -> Vec20s:
    return v


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)
    ret = identity(Vec20s(1))
    print("20dim vec:", ret)


if __name__ == "__main__":
    main()
