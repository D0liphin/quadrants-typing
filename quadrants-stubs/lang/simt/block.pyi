from typing import Callable, Generic, overload

from quadrants import i32, u32
from quadrants._ops import Arch

from quadrants_typing._algebra import _Dim, _Num, _NumAlt
from quadrants_typing._arraylike import _ArrayLike
from quadrants_typing._dim import Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, Dim7, Dim8, DimAny

class SharedArray(_ArrayLike[_Num, _Dim], Generic[_Num, _Dim]):
    # `dtype` and `shape` are inherited from `_ArrayLike`.
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim1], shape: tuple[int], dtype: type[_NumAlt], /
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim2], shape: tuple[int, int], dtype: type[_NumAlt], /
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim3], shape: tuple[int, int, int], dtype: type[_NumAlt], /
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim4],
        shape: tuple[int, int, int, int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim5],
        shape: tuple[int, int, int, int, int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim6],
        shape: tuple[int, int, int, int, int, int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim7],
        shape: tuple[int, int, int, int, int, int, int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, Dim8],
        shape: tuple[int, int, int, int, int, int, int, int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: SharedArray[_NumAlt, DimAny],
        shape: int | tuple[int, ...] | list[int],
        dtype: type[_NumAlt],
        /,
    ) -> None: ...

# barriers / fences
def sync() -> None: ...
def mem_fence() -> None: ...
def mem_sync() -> None: ...  # deprecated alias of `mem_fence`

# fused barrier + predicate reductions
def sync_all_nonzero(predicate: i32) -> i32: ...
def sync_any_nonzero(predicate: i32) -> i32: ...
def sync_count_nonzero(predicate: i32) -> i32: ...

# thread indices
def thread_idx() -> i32: ...
def global_thread_idx() -> i32: ...
def arch_uses_spv(arch: Arch) -> bool: ...

# block reductions: result dtype matches `value`; `op` is a `@qd.func` monoid.
def reduce(
    value: _Num, block_dim: int, op: Callable[[_Num, _Num], _Num], dtype: type[_Num]
) -> _Num: ...
def reduce_all(
    value: _Num, block_dim: int, op: Callable[[_Num, _Num], _Num], dtype: type[_Num]
) -> _Num: ...
def reduce_add(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def reduce_min(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def reduce_max(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def reduce_all_add(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def reduce_all_min(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def reduce_all_max(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...

# block scans: `exclusive_scan` takes an explicit `identity` (thread 0's value).
def inclusive_scan(
    value: _Num, block_dim: int, op: Callable[[_Num, _Num], _Num], dtype: type[_Num]
) -> _Num: ...
def exclusive_scan(
    value: _Num,
    block_dim: int,
    op: Callable[[_Num, _Num], _Num],
    identity: _Num,
    dtype: type[_Num],
) -> _Num: ...
def inclusive_add(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def inclusive_min(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def inclusive_max(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def exclusive_add(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def exclusive_min(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...
def exclusive_max(value: _Num, block_dim: int, dtype: type[_Num]) -> _Num: ...

# `bins` / `excl_prefix` are caller-allocated `SharedArray[i32]` out-params.
def radix_rank_match_atomic_or(
    key: u32,
    block_dim: int,
    radix_bits: int,
    bit_start: int,
    num_bits: int,
    bins: SharedArray[i32, DimAny],
    excl_prefix: SharedArray[i32, DimAny],
) -> i32: ...
