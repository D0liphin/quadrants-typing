from typing import TypeVar

import quadrants as qd

from quadrants_typing.annotations import _resolve_at_runtime


class NDArray:
    def __class_getitem__(cls, args):
        assert isinstance(args, tuple)
        assert len(args) == 2
        # NDArray is happily generic over ndim/dtype, so we can allow type
        # vars here, or DimAny (None at runtime), or even a typevar in the
        # dim case as well
        args0, args1 = _resolve_at_runtime(args[0]), _resolve_at_runtime(args[1])
        t = args[0] if not isinstance(args0, TypeVar) else None
        d = args[1] if isinstance(args1, int) else None
        return qd.types.ndarray(dtype=t, ndim=d)
