from typing import Annotated, get_args, get_origin


class Runtime:
    """`Runtime[TypeChecker, runtime]` — an annotation that reads as `TypeChecker`
    to a type checker but collapses to `runtime` when quadrants inspects it.

    Use very sparingly; you almost certainly do not need this. It exists for the
    handful of cases the static types can't express, e.g. a vector longer than
    the pre-declared dimensions (`qdt.Vec[qd.i16, Runtime[qdt.DimAny, 20]]`) or a
    plain scalar whose checker type is a restricted `Literal` set while its
    runtime dtype is a builtin (`Runtime[Literal[0, 1, 2], int]`).

    At runtime the subscription resolves eagerly to `runtime`, so an alias built
    with it (e.g. `Channel = Runtime[Literal[0, 1, 2], int]`) can be used directly
    as a kernel/func argument annotation and quadrants sees the resolved dtype.
    The type-checker view lives in `annotations.pyi`, where `Runtime` is
    `typing.Annotated` so `Runtime[TypeChecker, runtime]` is seen as `TypeChecker`.

    Note: a lazy PEP 695 alias (`type Channel = Runtime[...]`) stays wrapped at
    runtime, so use a plain assignment when the alias must resolve for quadrants.
    """

    def __class_getitem__(cls, args):
        if not isinstance(args, tuple) or len(args) != 2:
            raise TypeError("provide exactly one runtime value for this type annotation")
        return args[1]


def _resolve_at_runtime(ann):
    # `Runtime[...]` already collapsed to its runtime value before we get here;
    # this still unwraps a bare `typing.Annotated[T, runtime]` for good measure.
    if get_origin(ann) is Annotated:
        args = get_args(ann)
        if len(args) != 2:
            raise TypeError("provide exactly one runtime value for this type annotation")
        return args[1]
    return ann
