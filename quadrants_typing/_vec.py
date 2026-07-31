import quadrants as qd

from quadrants_typing.annotations import _resolve_at_runtime


class Vec:
    def __class_getitem__(cls, args):
        assert isinstance(args, tuple)
        assert len(args) == 2
        return qd.types.vector(_resolve_at_runtime(args[1]), _resolve_at_runtime(args[0]))
        # vector() needs (ndim, dtype), but we take [dtype, ndim]
