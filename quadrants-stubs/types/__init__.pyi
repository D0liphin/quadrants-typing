from typing import Any

from quadrants.types.primitive_types import f16 as f16
from quadrants.types.primitive_types import f32 as f32
from quadrants.types.primitive_types import f64 as f64
from quadrants.types.primitive_types import i8 as i8
from quadrants.types.primitive_types import i16 as i16
from quadrants.types.primitive_types import i32 as i32
from quadrants.types.primitive_types import i64 as i64
from quadrants.types.primitive_types import u1 as u1
from quadrants.types.primitive_types import u8 as u8
from quadrants.types.primitive_types import u16 as u16
from quadrants.types.primitive_types import u32 as u32
from quadrants.types.primitive_types import u64 as u64

import quadrants_typing as qdt
from quadrants_typing._algebra import _Num

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
