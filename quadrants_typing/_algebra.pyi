from typing import Literal, Self, TypeVar

from quadrants.types.primitive_types import (
    f16,
    f64,
    i8,
    i16,
    i64,
    u1,
    u8,
    u16,
    u32,
    u64,
)

from quadrants_typing._dim import DimAny

# --- primitive type hierarchy (nominal cores) ---
#
# `i32`/`f32` collapse to the builtins `int`/`float` (see INT_FLOAT_ALIASING.md),
# so a dtype "family" is a *union* of the builtin with the nominal core that the
# remaining ten dtypes derive from. The nominal cores are private (`_`-prefixed);
# the public `Number`/`Integer`/`Floating` are the unions and are what you use as
# TypeVar bounds.

class DType: ...

class Number(DType):
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: Self, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    def __truediv__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...
    def __floordiv__(self, other: Self, /) -> Self: ...
    def __rfloordiv__(self, other: Self, /) -> Self: ...
    def __mod__(self, other: Self, /) -> Self: ...
    def __rmod__(self, other: Self, /) -> Self: ...
    def __pow__(self, other: Self, /) -> Self: ...
    def __rpow__(self, other: Self, /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...

class Floating(Number):
    """Floating-point nominal dtype. Mirrors :class:`numpy.floating`."""
    def __init__(self, _: float, /) -> None: ...
    def __float__(self) -> float: ...

class Integer(Number):
    """Integer nominal dtype. Mirrors :class:`numpy.integer`."""
    def __init__(self, _: int, /) -> None: ...
    def __index__(self) -> int: ...
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __and__(self, other: Self, /) -> Self: ...
    def __rand__(self, other: Self, /) -> Self: ...
    def __or__(self, other: Self, /) -> Self: ...
    def __ror__(self, other: Self, /) -> Self: ...
    def __xor__(self, other: Self, /) -> Self: ...
    def __rxor__(self, other: Self, /) -> Self: ...
    def __lshift__(self, other: Self, /) -> Self: ...
    def __rlshift__(self, other: Self, /) -> Self: ...
    def __rshift__(self, other: Self, /) -> Self: ...
    def __rrshift__(self, other: Self, /) -> Self: ...
    def __invert__(self) -> Self: ...

class SignedInteger(Integer):
    """Signed integer quadrants dtype. Mirrors :class:`numpy.signedinteger`."""

class UnsignedInteger(Integer):
    """Unsigned integer quadrants dtype. Mirrors :class:`numpy.unsignedinteger`."""

# --- initializer unions (shared by the vector/matrix constructors) ---
#
# These stay pinned to the nominal cores plus the builtins: a `float` fill covers
# `f32`, an `int` fill covers `i32`, and the nominal cores cover the rest.

type _FloatInit = float | Floating | Integer
type _IntegerInit = int | Integer

# Container constructors accept *any* numeric value regardless of the element
# type: the element cast is implicit (a `Mat[f32, N, M]` is initializable from
# ints, `Vec[i8, D]` from floats, etc.). `Number` covers every nominal core.
type _NumInit = int | float | Number

# --- shared type variables ---------------------------------------------------
#
# The public numeric surface (`Vec`/`Mat` elements, every `math`/`_ops` function)
# is generic over `_Num`: a *constrained* TypeVar enumerating every concrete
# dtype. A constraint set (unlike an abstract bound) forbids union elements such
# as `Vec[int | float, D]` structurally, and — because the members are leaf
# types — never widens a nominal element (`Vec[f16, D]` stays `f16`). `int`/
# `float` are the builtins onto which `i32`/`f32` collapse; the ten nominal cores
# cover the rest.
_Num = TypeVar("_Num", int, float, f16, f64, i8, i16, i64, u1, u8, u16, u32, u64)

# A second independent copy, for signatures with two unrelated numeric slots
# (e.g. `cast`'s source vs target dtype, `select`'s condition vs value).
_Num2 = TypeVar("_Num2", int, float, f16, f64, i8, i16, i64, u1, u8, u16, u32, u64)

# Family subsets, for overloads that must split integer- vs float-typed results
# (e.g. `Mat.to_list`, integer-only ops). Same enumeration, partitioned. These
# need no method-scoped twin (see below): no container is generic over a family
# subset, so they are never class-scoped and are legal in `self` annotations.
_Int = TypeVar("_Int", int, i8, i16, i64, u1, u8, u16, u32, u64)
_Float = TypeVar("_Float", float, f16, f64)

# `NDArray` elements are *bounded*, not enumerated: an element may itself be a
# `Vec`/`Mat` (a `DType` subclass that can't be listed), so a bound is required.
# A bound still solves to the actual argument, so no widening occurs.
_El = TypeVar("_El", bound="DType | int | float")

# Dimension markers, shared across the container generics and their functions.
_Dim = TypeVar("_Dim", bound=DimAny, covariant=True)
_Row = TypeVar("_Row", bound=DimAny, covariant=True)
_Col = TypeVar("_Col", bound=DimAny, covariant=True)

# Alternate element/dimension markers, for `self` annotations in `__init__`:
# pyright rejects class-scoped type variables there specifically
# (`reportInvalidTypeVarUse`), so a `self: Vec[_NumAlt, _DimAlt]` overload needs
# method-scoped copies. Ordinary methods may use `_Num`/`_Dim` in `self` freely,
# as `Vec.x` and `Mat.__matmul__` do. Same enumerations/bounds; distinct
# identities.
_NumAlt = TypeVar("_NumAlt", int, float, f16, f64, i8, i16, i64, u1, u8, u16, u32, u64)
_DimAlt = TypeVar("_DimAlt", bound=DimAny, covariant=True)
_RowAlt = TypeVar("_RowAlt", bound=DimAny, covariant=True)
_ColAlt = TypeVar("_ColAlt", bound=DimAny, covariant=True)

# Compile-time constants for the `simt.subgroup` tiled ops. `log2_size` picks a
# tile of `2**log2_size` lanes: `0` (a tile of one, a no-op / identity sort) up
# to `5` (32-lane wave32) or `6` (64-lane AMDGPU wave64). `ballot_first_n`'s `n`
# is a lane count in `[1, 32]`. These are `template()` args, so they must be
# compile-time literals — the type is a hint, not enforcement of static-ness.
type _Log2Size = Literal[0, 1, 2, 3, 4, 5, 6]
type _BallotWidth = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
