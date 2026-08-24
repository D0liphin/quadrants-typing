# pyright: strict
"""Define a `@qd.dataclass` struct, build it via its synthesized constructor, and use a member func."""

import quadrants as qd

import quadrants_typing as qdt


# A quadrants struct: a 3-D position and a scalar mass. `@qd.dataclass`
# synthesizes a constructor from these field annotations, just like
# `dataclasses.dataclass`.
@qd.dataclass
class Particle:
    pos: qdt.Vec[qd.f32, qdt.Dim3]
    mass: qd.f32

    # Member functions defined on the struct operate on an instance.
    @qd.func
    def momentum(self, velocity: qdt.Vec[qd.f32, qdt.Dim3]) -> qdt.Vec[qd.f32, qdt.Dim3]:
        return velocity * self.mass


# Take a particle by value and return the magnitude of its momentum along +z.
@qd.kernel
def momentum_norm(p: Particle) -> qd.f32:
    v = qd.math.vec3(0.0, 0.0, 1.0)
    return p.momentum(v).norm()


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Build a particle through the synthesized constructor (positional or keyword).
    p = Particle(pos=qd.math.vec3(1.0), mass=3.0)

    print("particle:", p)
    print("momentum norm:", momentum_norm(p))


if __name__ == "__main__":
    main()
