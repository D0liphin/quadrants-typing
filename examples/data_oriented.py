# pyright: strict
"""Group an ndarray field and a member kernel in a `@qd.data_oriented` class."""

import quadrants as qd


# A class that owns its own array and the kernel that operates on it.
@qd.data_oriented
class Counter:
    def __init__(self, n: int) -> None:
        # Allocate the array this instance operates on.
        self.values = qd.ndarray(qd.f32, (n,))

    # Add one to every element of the instance's array.
    @qd.kernel
    def increment(self) -> None:
        for i in range(self.values.shape[0]):
            self.values[i] = self.values[i] + 1.0


def main() -> None:
    # Select the CPU backend for compilation.
    qd.init(arch=qd.cpu)

    # Create an instance and step it twice.
    counter = Counter(5)
    counter.increment()
    counter.increment()

    print(counter.values.to_numpy())


if __name__ == "__main__":
    main()
