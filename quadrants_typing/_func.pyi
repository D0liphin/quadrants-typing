from typing import Any, Concatenate, overload

class Func[**P, R]:
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @overload
    def __get__(self, instance: None, owner: Any = ..., /) -> Func[P, R]: ...
    @overload
    def __get__[S, **Q, R2](
        self: Func[Concatenate[S, Q], R2], instance: S, owner: Any = ..., /
    ) -> Func[Q, R2]: ...
