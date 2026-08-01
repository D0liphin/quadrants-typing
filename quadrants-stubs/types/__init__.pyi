from typing import Any

from quadrants.types.primitive_types import f16 as f16
from quadrants.types.primitive_types import f32 as f32
from quadrants.types.primitive_types import f64 as f64

# Long-form dtype spellings (NumPy-style). Each is the same object as its terse
# alias — see `primitive_types.pyi`.
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
from quadrants_typing._algebra import _Num

# dtype predicates
def is_integral(dtype: type[_Num], /) -> bool: ...
def is_real(dtype: type[_Num], /) -> bool: ...
def is_signed(dtype: type[_Num], /) -> bool: ...
def is_tensor(dtype: type[_Num], /) -> bool: ...

# vector
def vector(dim: int, dtype: type[_Num]) -> type[qdt.Vec[_Num, qdt.DimAny]]: ...

# matrix
def matrix(
    row: int,
    col: int,
    dtype: type[_Num],
) -> type[qdt.Mat[_Num, qdt.DimAny, qdt.DimAny]]: ...

# ignore
def __getattr__(name: str) -> Any: ...
