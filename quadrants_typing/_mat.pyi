from typing import Any, Generic, Self, overload

import numpy as np
import numpy.typing as npt
import quadrants as qd

from quadrants_typing._algebra import (
    DType,
    _Col,
    _ColAlt,
    _FloatNum,
    _IntNum,
    _Num,
    _NumAlt,
    _NumInit,
    _Row,
    _RowAlt,
)
from quadrants_typing._dim import Dim2, Dim3, DimAny
from quadrants_typing._vec import Vec

type _MatIndex = tuple[int, int]

class Mat(DType, Generic[_Num, _Row, _Col]):
    """A matrix with ``_Row`` rows and ``_Col`` columns.

    A new type can be created as below:

    ```
    Mat4x3f = Mat[f32, Dim4, Dim3]
    ```

    But you probably want to use an existing type e.g. `Mat3f`
    """

    n: int
    """Class attribute only! Number of rows"""
    m: int
    """Class attribute only! Number of columns"""

    # Any numeric value initializes any matrix — the element cast is implicit —
    # so every arg takes `_NumInit`. Overloads split only on shape / arg form.
    # `self` uses method-scoped `_NumAlt`/`_RowAlt`/`_ColAlt` (class-scoped vars
    # are barred from `self` annotations).
    @overload
    def __init__(
        self: Mat[_NumAlt, Dim2, Dim2],
        row0: tuple[_NumInit, _NumInit, _NumInit],
        row1: tuple[_NumInit, _NumInit, _NumInit],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: Mat[_NumAlt, Dim3, Dim3],
        row0: tuple[_NumInit, _NumInit, _NumInit],
        row1: tuple[_NumInit, _NumInit, _NumInit],
        row2: tuple[_NumInit, _NumInit, _NumInit],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: Mat[_NumAlt, _RowAlt, _ColAlt],
        rows: list[list[_NumInit]],
        /,
    ) -> None: ...
    @overload
    def __init__(
        self: Mat[_NumAlt, _RowAlt, _ColAlt],
        rows: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
        /,
    ) -> None: ...
    def __getitem__(self, key: _MatIndex, /) -> _Num: ...
    def __setitem__(self, key: _MatIndex, value: _Num, /) -> None: ...
    def __len__(self) -> int: ...
    def __neg__(self) -> Self: ...
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: _Num, /) -> Self: ...
    def __rmul__(self, other: _Num, /) -> Self: ...
    def __truediv__(self, other: _Num, /) -> Self: ...
    @overload
    def __matmul__[D_: DimAny](
        self: Mat[_Num, D_, D_], other: Mat[_Num, D_, D_], /
    ) -> Mat[_Num, D_, D_]: ...
    @overload
    def __matmul__[D_: DimAny](
        self: Mat[_Num, D_, D_], other: Vec[_Num, D_], /
    ) -> Vec[_Num, D_]: ...
    def transpose(self) -> Self: ...
    def determinant(self) -> _Num: ...
    def trace(self) -> _Num: ...
    def inverse(self) -> Self: ...
    def frobenius_inner(self, other: Self) -> _Num: ...
    @overload
    def to_list(self: Mat[_FloatNum, DimAny, DimAny]) -> list[list[float]]: ...
    @overload
    def to_list(self: Mat[_IntNum, DimAny, DimAny]) -> list[list[int]]: ...
    # `to_numpy` maps each concrete element dtype to its numpy counterpart
    # (mirrors `NDArray.to_numpy`). `qd.i32`/`qd.f32` are the builtins `int`/
    # `float`; the ten nominal cores map by width.
    @overload
    def to_numpy(self: Mat[qd.i8, DimAny, DimAny]) -> npt.NDArray[np.int8]: ...
    @overload
    def to_numpy(self: Mat[qd.i16, DimAny, DimAny]) -> npt.NDArray[np.int16]: ...
    @overload
    def to_numpy(self: Mat[qd.i32, DimAny, DimAny]) -> npt.NDArray[np.int32]: ...
    @overload
    def to_numpy(self: Mat[qd.i64, DimAny, DimAny]) -> npt.NDArray[np.int64]: ...
    @overload
    def to_numpy(self: Mat[qd.u1, DimAny, DimAny]) -> npt.NDArray[np.bool_]: ...
    @overload
    def to_numpy(self: Mat[qd.u8, DimAny, DimAny]) -> npt.NDArray[np.uint8]: ...
    @overload
    def to_numpy(self: Mat[qd.u16, DimAny, DimAny]) -> npt.NDArray[np.uint16]: ...
    @overload
    def to_numpy(self: Mat[qd.u32, DimAny, DimAny]) -> npt.NDArray[np.uint32]: ...
    @overload
    def to_numpy(self: Mat[qd.u64, DimAny, DimAny]) -> npt.NDArray[np.uint64]: ...
    @overload
    def to_numpy(self: Mat[qd.f16, DimAny, DimAny]) -> npt.NDArray[np.float16]: ...
    @overload
    def to_numpy(self: Mat[qd.f32, DimAny, DimAny]) -> npt.NDArray[np.float32]: ...
    @overload
    def to_numpy(self: Mat[qd.f64, DimAny, DimAny]) -> npt.NDArray[np.float64]: ...
