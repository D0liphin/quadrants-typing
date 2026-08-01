"""Type annotations for quadrants values.

Every public name here is a real annotation you can write in source, and each
resolves to a concrete quadrants runtime object where quadrants' compiler needs
one (e.g. `Vec3f`, `Template[qd.i32]`, `NDArray[qd.f32, DimAny]`).

The abstractions `DType`, `Number`, `Floating`, `Integer`, `SignedInteger`,
`UnsignedInteger`, and `NumericOps` are annotation-only: they name the dtype
family for the type checker and have no runtime object (accessing them at
runtime raises `AttributeError`).
"""

from typing import TYPE_CHECKING

from quadrants_typing._aliases import (
    Mat2f,
    Mat2i,
    Mat3f,
    Mat3i,
    Mat4f,
    Mat4i,
    Vec2f,
    Vec2i,
    Vec3f,
    Vec3i,
    Vec4f,
    Vec4i,
)
from quadrants_typing._dim import (
    Dim1,
    Dim2,
    Dim3,
    Dim4,
    Dim5,
    Dim6,
    Dim7,
    Dim8,
    DimAny,
    dim,
)
from quadrants_typing._mat import Mat
from quadrants_typing._ndarray import NDArray
from quadrants_typing._template import Template
from quadrants_typing._vec import Vec
from quadrants_typing._primitive_types import uint
from quadrants_typing import annotations

if TYPE_CHECKING:
    from quadrants_typing._algebra import (
        DType,
        Floating,
        Integer,
        Number,
        SignedInteger,
        UnsignedInteger,
    )
    from quadrants_typing._func import Func
    from quadrants_typing._kernel import Kernel
    from quadrants_typing._dim import DimConcrete

__all__ = [
    "DType",
    "Dim1",
    "Dim2",
    "Dim3",
    "Dim4",
    "Dim5",
    "Dim6",
    "Dim7",
    "Dim8",
    "DimAny",
    "DimConcrete",
    "Floating",
    "Integer",
    "Mat",
    "Mat2f",
    "Mat2i",
    "Mat3f",
    "Mat3i",
    "Mat4f",
    "Mat4i",
    "NDArray",
    "Number",
    "SignedInteger",
    "UnsignedInteger",
    "uint",
    "Template",
    "Vec",
    "Vec2f",
    "Vec2i",
    "Vec3f",
    "Vec3i",
    "Vec4f",
    "Vec4i",
    "dim",
    "annotations",
    "Func",
    "Kernel",
]
