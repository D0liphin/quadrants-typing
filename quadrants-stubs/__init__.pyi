from typing import Any, Iterable, overload

from quadrants import interop as interop
from quadrants import math as math
from quadrants import types as types
from quadrants._ops import Arch as Arch
from quadrants._ops import Backend as Backend
from quadrants._ops import amdgpu as amdgpu
from quadrants._ops import arm64 as arm64
from quadrants._ops import atomic_add as atomic_add
from quadrants._ops import atomic_and as atomic_and
from quadrants._ops import atomic_cas as atomic_cas
from quadrants._ops import atomic_exchange as atomic_exchange
from quadrants._ops import atomic_max as atomic_max
from quadrants._ops import atomic_min as atomic_min
from quadrants._ops import atomic_mul as atomic_mul
from quadrants._ops import atomic_or as atomic_or
from quadrants._ops import atomic_sub as atomic_sub
from quadrants._ops import atomic_xor as atomic_xor
from quadrants._ops import bit_cast as bit_cast
from quadrants._ops import bit_shr as bit_shr
from quadrants._ops import cast as cast
from quadrants._ops import cpu as cpu
from quadrants._ops import cuda as cuda
from quadrants._ops import data_oriented as data_oriented
from quadrants._ops import func as func
from quadrants._ops import gpu as gpu
from quadrants._ops import grouped as grouped
from quadrants._ops import init as init
from quadrants._ops import kernel as kernel
from quadrants._ops import loop_config as loop_config
from quadrants._ops import metal as metal
from quadrants._ops import ndrange as ndrange
from quadrants._ops import python as python
from quadrants._ops import random as random
from quadrants._ops import raw_div as raw_div
from quadrants._ops import raw_mod as raw_mod
from quadrants._ops import reset as reset
from quadrants._ops import select as select
from quadrants._ops import sync as sync
from quadrants._ops import vulkan as vulkan
from quadrants._ops import x64 as x64
from quadrants._ops import x86_64 as x86_64
from quadrants.lang import simt as simt
from quadrants.math import acos as acos
from quadrants.math import asin as asin
from quadrants.math import atan2 as atan2
from quadrants.math import ceil as ceil
from quadrants.math import cos as cos
from quadrants.math import exp as exp
from quadrants.math import floor as floor
from quadrants.math import log as log
from quadrants.math import max as max
from quadrants.math import min as min
from quadrants.math import pow as pow
from quadrants.math import round as round
from quadrants.math import sin as sin
from quadrants.math import sqrt as sqrt
from quadrants.math import tan as tan
from quadrants.math import tanh as tanh
from quadrants.types.primitive_types import f16 as f16
from quadrants.types.primitive_types import f32 as f32
from quadrants.types.primitive_types import f64 as f64
from quadrants.types.primitive_types import float16 as float16
from quadrants.types.primitive_types import float32 as float32
from quadrants.types.primitive_types import float64 as float64
from quadrants.types.primitive_types import i8 as i8
from quadrants.types.primitive_types import i16 as i16
from quadrants.types.primitive_types import i32 as i32
from quadrants.types.primitive_types import i64 as i64
from quadrants.types.primitive_types import int8 as int8
from quadrants.types.primitive_types import int16 as int16
from quadrants.types.primitive_types import int32 as int32
from quadrants.types.primitive_types import int64 as int64
from quadrants.types.primitive_types import u1 as u1
from quadrants.types.primitive_types import u8 as u8
from quadrants.types.primitive_types import u16 as u16
from quadrants.types.primitive_types import u32 as u32
from quadrants.types.primitive_types import u64 as u64
from quadrants.types.primitive_types import uint1 as uint1
from quadrants.types.primitive_types import uint8 as uint8
from quadrants.types.primitive_types import uint16 as uint16
from quadrants.types.primitive_types import uint32 as uint32
from quadrants.types.primitive_types import uint64 as uint64

import quadrants_typing as qdt
from quadrants_typing._algebra import _Col, _Dim, _El, _Num, _Row

# `abs` is a top-level builtin-style op, not a `quadrants.math` member.
@overload
def abs(x: _Num) -> _Num: ...
@overload
def abs(x: qdt.Vec[_Num, _Dim]) -> qdt.Vec[_Num, _Dim]: ...
@overload
def abs(x: qdt.Mat[_Num, _Row, _Col]) -> qdt.Mat[_Num, _Row, _Col]: ...

# static

@overload
def static(x: bool, /) -> u1: ...
@overload
def static(x: int, /) -> i32: ...
@overload
def static(x: float, /) -> f32: ...
@overload
def static(x: Iterable[int], /) -> Iterable[i32]: ...
@overload
def static(x: Any, y: Any, /, *xs: Any) -> list[Any]: ...
@overload
def static[T](x: T, /) -> T: ...

# compile-time print / assert: evaluated during tracing, no runtime overhead.
def static_print(*args: Any, **kwargs: Any) -> None: ...
def static_assert(cond: bool, msg: str | None = ...) -> None: ...
def __getattr__(name: str) -> Any: ...

# ndarray

@overload
def ndarray(
    dtype: type[_El], shape: tuple[int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim1]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim2]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim3]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim4]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim5]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int, int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim6]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int, int, int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim7]: ...
@overload
def ndarray(
    dtype: type[_El], shape: tuple[int, int, int, int, int, int, int, int], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.Dim8]: ...
@overload
def ndarray(
    dtype: type[_El], shape: int | tuple[int, ...], needs_grad: bool = ...
) -> qdt.NDArray[_El, qdt.DimAny]: ...
