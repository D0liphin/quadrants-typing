import quadrants as qd

from quadrants_typing.annotations import _resolve_at_runtime


class Mat:
    def __class_getitem__(cls, args):
        assert isinstance(args, tuple)
        assert len(args) == 3
        return qd.types.matrix(
            _resolve_at_runtime(args[1]),
            _resolve_at_runtime(args[2]),
            _resolve_at_runtime(args[0]),
        )
        # matrix() needs (n_rows, n_cols, dtype), but we take
        # [dtype, rowD, colD] with dtype first to match Vec/NDArray
