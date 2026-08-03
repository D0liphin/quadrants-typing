from typing import Any, Concatenate, overload

class Kernel[**P, R]:
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @overload
    def __get__(self, instance: None, owner: Any = ..., /) -> Kernel[P, R]: ...
    @overload
    def __get__[S, **Q, R2](
        self: Kernel[Concatenate[S, Q], R2], instance: S, owner: Any = ..., /
    ) -> Kernel[Q, R2]: ...
