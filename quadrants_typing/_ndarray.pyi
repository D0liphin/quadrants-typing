from typing import Any, Generic, Self, overload

import numpy as np
import numpy.typing as npt
import quadrants as qd
import torch

from quadrants_typing._algebra import (
    Floating,
    Integer,
    Number,
    _Dim,
    _El,
)
from quadrants_typing._arraylike import _ArrayLike
from quadrants_typing._dim import DimAny
from quadrants_typing._mat import Mat
from quadrants_typing._vec import Vec

class _ElementShape:
    @overload
    def __get__(self, instance: NDArray[Number, Any], owner: Any = None) -> tuple[()]: ...
    @overload
    def __get__(self, instance: NDArray[Vec[Any, Any], Any], owner: Any = None) -> tuple[int,]: ...
    @overload
    def __get__(
        self, instance: NDArray[Mat[Any, Any, Any], Any], owner: Any = None
    ) -> tuple[int, int]: ...

class NDArray(_ArrayLike[_El, _Dim], Generic[_El, _Dim]):
    """Type annotation for kernel ndarray arguments.

    You can pass tensors to kernels that take this as the argument. This is a
    special type that is used internally by quadrants to compile the kernels
    with type information.

    `_El` is also happily a type var!, and `_Dim` is also happily `DimAny`!

    Use `len(ndarray.shape)` for `ndarray.ndim`!
    """

    # `dtype` and `shape` are inherited from `_ArrayLike`.
    element_shape = _ElementShape()

    @overload
    def fill[T_: Integer | int](self: NDArray[T_, DimAny], value: int | T_) -> None: ...
    @overload
    def fill[T_: Floating | float](self: NDArray[T_, DimAny], value: float | T_) -> None: ...
    def from_torch(self, tensor: torch.Tensor) -> None: ...
    def to_torch(self) -> torch.Tensor: ...
    @overload
    def from_numpy(
        self: NDArray[qd.i8, DimAny]
        | NDArray[Vec[qd.i8, Any], DimAny]
        | NDArray[Mat[qd.i8, Any, Any], DimAny],
        ndarray: npt.NDArray[np.int8],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.i16, DimAny]
        | NDArray[Vec[qd.i16, Any], DimAny]
        | NDArray[Mat[qd.i16, Any, Any], DimAny],
        ndarray: npt.NDArray[np.int16],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.i32, DimAny]
        | NDArray[Vec[qd.i32, Any], DimAny]
        | NDArray[Mat[qd.i32, Any, Any], DimAny],
        ndarray: npt.NDArray[np.int32],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.i64, DimAny]
        | NDArray[Vec[qd.i64, Any], DimAny]
        | NDArray[Mat[qd.i64, Any, Any], DimAny],
        ndarray: npt.NDArray[np.int64],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.u1, DimAny]
        | NDArray[Vec[qd.u1, Any], DimAny]
        | NDArray[Mat[qd.u1, Any, Any], DimAny],
        ndarray: npt.NDArray[np.bool_],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.u8, DimAny]
        | NDArray[Vec[qd.u8, Any], DimAny]
        | NDArray[Mat[qd.u8, Any, Any], DimAny],
        ndarray: npt.NDArray[np.uint8],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.u16, DimAny]
        | NDArray[Vec[qd.u16, Any], DimAny]
        | NDArray[Mat[qd.u16, Any, Any], DimAny],
        ndarray: npt.NDArray[np.uint16],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.u32, DimAny]
        | NDArray[Vec[qd.u32, Any], DimAny]
        | NDArray[Mat[qd.u32, Any, Any], DimAny],
        ndarray: npt.NDArray[np.uint32],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.u64, DimAny]
        | NDArray[Vec[qd.u64, Any], DimAny]
        | NDArray[Mat[qd.u64, Any, Any], DimAny],
        ndarray: npt.NDArray[np.uint64],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.f16, DimAny]
        | NDArray[Vec[qd.f16, Any], DimAny]
        | NDArray[Mat[qd.f16, Any, Any], DimAny],
        ndarray: npt.NDArray[np.float16],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.f32, DimAny]
        | NDArray[Vec[qd.f32, Any], DimAny]
        | NDArray[Mat[qd.f32, Any, Any], DimAny],
        ndarray: npt.NDArray[np.float32],
    ) -> None: ...
    @overload
    def from_numpy(
        self: NDArray[qd.f64, DimAny]
        | NDArray[Vec[qd.f64, Any], DimAny]
        | NDArray[Mat[qd.f64, Any, Any], DimAny],
        ndarray: npt.NDArray[np.float64],
    ) -> None: ...
    @overload
    def to_numpy(
        self: NDArray[qd.i8, DimAny]
        | NDArray[Vec[qd.i8, Any], DimAny]
        | NDArray[Mat[qd.i8, Any, Any], DimAny],
    ) -> npt.NDArray[np.int8]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.i16, DimAny]
        | NDArray[Vec[qd.i16, Any], DimAny]
        | NDArray[Mat[qd.i16, Any, Any], DimAny],
    ) -> npt.NDArray[np.int16]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.i32, DimAny]
        | NDArray[Vec[qd.i32, Any], DimAny]
        | NDArray[Mat[qd.i32, Any, Any], DimAny],
    ) -> npt.NDArray[np.int32]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.i64, DimAny]
        | NDArray[Vec[qd.i64, Any], DimAny]
        | NDArray[Mat[qd.i64, Any, Any], DimAny],
    ) -> npt.NDArray[np.int64]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.u1, DimAny]
        | NDArray[Vec[qd.u1, Any], DimAny]
        | NDArray[Mat[qd.u1, Any, Any], DimAny],
    ) -> npt.NDArray[np.bool_]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.u8, DimAny]
        | NDArray[Vec[qd.u8, Any], DimAny]
        | NDArray[Mat[qd.u8, Any, Any], DimAny],
    ) -> npt.NDArray[np.uint8]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.u16, DimAny]
        | NDArray[Vec[qd.u16, Any], DimAny]
        | NDArray[Mat[qd.u16, Any, Any], DimAny],
    ) -> npt.NDArray[np.uint16]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.u32, DimAny]
        | NDArray[Vec[qd.u32, Any], DimAny]
        | NDArray[Mat[qd.u32, Any, Any], DimAny],
    ) -> npt.NDArray[np.uint32]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.u64, DimAny]
        | NDArray[Vec[qd.u64, Any], DimAny]
        | NDArray[Mat[qd.u64, Any, Any], DimAny],
    ) -> npt.NDArray[np.uint64]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.f16, DimAny]
        | NDArray[Vec[qd.f16, Any], DimAny]
        | NDArray[Mat[qd.f16, Any, Any], DimAny],
    ) -> npt.NDArray[np.float16]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.f32, DimAny]
        | NDArray[Vec[qd.f32, Any], DimAny]
        | NDArray[Mat[qd.f32, Any, Any], DimAny],
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def to_numpy(
        self: NDArray[qd.f64, DimAny]
        | NDArray[Vec[qd.f64, Any], DimAny]
        | NDArray[Mat[qd.f64, Any, Any], DimAny],
    ) -> npt.NDArray[np.float64]: ...
    # `__getitem__` / `__setitem__` are inherited from `_ArrayLike`.
    def __add__(self, other: _El, /) -> Self: ...
    def __radd__(self, other: _El, /) -> Self: ...
    def __sub__(self, other: _El, /) -> Self: ...
    def __rsub__(self, other: _El, /) -> Self: ...
    def __mul__(self, other: _El, /) -> Self: ...
    def __rmul__(self, other: _El, /) -> Self: ...
    def __truediv__(self, other: _El, /) -> Self: ...
    def __rtruediv__(self, other: _El, /) -> Self: ...
