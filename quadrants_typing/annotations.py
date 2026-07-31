from typing import Annotated, get_args, get_origin

Runtime = Annotated
"""
Allows you to specify what this annotation resolves to at runtime, use very
sparingly. You almost certainly do not need this.
"""


def _resolve_at_runtime(ann):
    if get_origin(ann) is Annotated:
        args = get_args(ann)
        if len(args) != 2:
            raise TypeError("provide exactly one runtime value for this type annotation")
        return get_args(ann)[1]
    return ann
