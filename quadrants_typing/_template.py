from typing import TYPE_CHECKING, TypeVar

import quadrants as qd

if TYPE_CHECKING:
    _T = TypeVar("_T")
    type _Template[T] = T
    Template = _Template[_T]
    """
    A value known at compile time. Use the parameter in `qd.static()` to have
    quadrants dispatch e.g. a branch or unroll a loop at compile time.

    ```
    @qd.kernel
    def identity_tp(x: Template[i32]) -> i32:
        return x
    ```

    Each call of `identity_tp` makes a new version specialized to the particular
    *value* of the integer passed in.

    You can imagine this something like this:

    ```
    def identity_tp(x: Template[i32]) -> i32:
        lookup = {}
        if x in lookup:
            return lookup[x]()
        else:
            lookup[x] = _compile_new_function()
            return lookup[x]()
    ```

    The function is defined as a single instruction, but requires an additional
    compile the first time it is called with a particular set of args.

    For highly variable arguments, the space of functions is likely to explode
    beyond any reasonable benefit, and templates ought not be used.
    """
else:

    class Template:
        def __class_getitem__(cls, arg):
            return qd.template()
