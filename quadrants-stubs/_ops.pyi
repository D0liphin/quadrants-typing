from enum import IntEnum
from typing import Any, Callable, Iterable, Iterator, TypeVar, overload

from quadrants.types.primitive_types import f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

import quadrants_typing as qdt
from quadrants_typing._algebra import _Col, _Dim, _El, _Num, _Num2, _Row

_Vec2i = qdt.Vec[i32, qdt.Dim2]
_Vec3i = qdt.Vec[i32, qdt.Dim3]

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])

# --- backends ---

class Arch:
    """Compilation target (e.g. `qd.cpu`, `qd.cuda`)."""

cpu: Arch
cuda: Arch
metal: Arch
vulkan: Arch
amdgpu: Arch
x64: Arch
x86_64: Arch
arm64: Arch
python: Arch
gpu: list[Arch]

class Backend(IntEnum):
    FIELD = 0
    NDARRAY = 1

def init(arch: Arch | list[Arch] | None = None, **kwargs: Any) -> None: ...
def reset() -> None: ...

# --- kernels and functions ---

class _FuncDecorator:
    def __call__[**P, R](self, callable: Callable[P, R]) -> qdt.Func[P, R]: ...

@overload
def func[**P, R](fn: Callable[P, R], /) -> qdt.Func[P, R]: ...
@overload
def func(*, is_real_function: bool = ..., requires_top_level: bool = ...) -> _FuncDecorator: ...

class _KernelDecorator:
    def __call__[**P, R](self, callable: Callable[P, R]) -> qdt.Kernel[P, R]: ...

@overload
def kernel[**P, R](fn: Callable[P, R], /) -> qdt.Kernel[P, R]: ...
@overload
def kernel(
    *, pure: bool = ..., fastcache: bool = ..., graph: bool = ..., checkpoints: bool = ...
) -> _KernelDecorator: ...

# data oriented
@overload
def data_oriented(cls: type[_T], /) -> type[_T]: ...
@overload
def data_oriented(*, template_primitives: bool = ...) -> Callable[[type[_T]], type[_T]]: ...

# dataclass: turns an annotated class into a quadrants struct type. Unlike
# `data_oriented`, it takes no options, so it is only ever applied bare.
def dataclass(cls: type[_T], /) -> type[_T]: ...
def sync() -> None: ...

# --- iteration ---

@overload
def ndrange(
    r0: tuple[int, int] | int,
    /,
) -> Iterable[tuple[int]]: ...
@overload
def ndrange(
    r0: tuple[int, int] | int,
    r1: tuple[int, int] | int,
    /,
) -> Iterable[_Vec2i]: ...
@overload
def ndrange(
    r0: tuple[int, int] | int,
    r1: tuple[int, int] | int,
    r2: tuple[int, int] | int,
    /,
) -> Iterable[_Vec3i]: ...
@overload
def grouped(
    ndarray: qdt.NDArray[_El, _Dim],
) -> Iterator[qdt.Vec[int, _Dim]]: ...
@overload
def grouped(it: Iterable[_Vec3i], /) -> Iterator[_Vec3i]: ...
def loop_config(
    *,
    block_dim: int | None = None,
    serialize: bool = False,
    parallelize: int | None = None,
    block_dim_adaptive: bool = True,
    bit_vectorize: bool = False,
    name: str | None = None,
) -> None: ...

# --- casts ---

@overload
def cast(x: qdt.Vec[_Num, _Dim], ty: type[_Num2]) -> qdt.Vec[_Num2, _Dim]: ...
@overload
def cast(x: qdt.Mat[_Num, _Row, _Col], ty: type[_Num2]) -> qdt.Mat[_Num2, _Row, _Col]: ...
@overload
def cast(x: _Num, ty: type[_Num2]) -> _Num2: ...

# bit cast
@overload
def bit_cast[From: (u8, i8), To: (u8, i8)](obj: From, dtype: type[To]) -> To: ...
@overload
def bit_cast[From: (u16, i16, f16), To: (u16, i16, f16)](obj: From, dtype: type[To]) -> To: ...
@overload
def bit_cast[From: (u32, i32, f32), To: (u32, i32, f32)](obj: From, dtype: type[To]) -> To: ...
@overload
def bit_cast[From: (u64, i64, f64), To: (u64, i64, f64)](obj: From, dtype: type[To]) -> To: ...

# --- integer / raw arithmetic ---

def raw_div(x1: _Num, x2: _Num) -> _Num: ...
def raw_mod(x1: _Num, x2: _Num) -> _Num: ...
def bit_shr(x1: _Num, x2: _Num) -> _Num: ...
def select(cond: _Num2, x1: _Num, x2: _Num) -> _Num: ...

# --- atomics ---

def atomic_add[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_sub[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_mul[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_max[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_min[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_and[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_or[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_xor[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_exchange[T: (int, float)](dst: T, value: T, /) -> T: ...
def atomic_cas[T: (int, float)](dst: T, expected: T, desired: T, /) -> T: ...

# --- randomness ---

def random(dtype: type[_Num]) -> _Num: ...
