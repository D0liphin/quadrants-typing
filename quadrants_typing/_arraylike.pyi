from typing import Any, Generic, overload

from quadrants_typing._algebra import _Dim, _El
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
)
from quadrants_typing._vec import Vec

_Vec2i = Vec[int, Dim2]
_Vec3i = Vec[int, Dim3]
_Vec4i = Vec[int, Dim4]
_Vec5i = Vec[int, Dim5]
_Vec6i = Vec[int, Dim6]
_Vec7i = Vec[int, Dim7]
_Vec8i = Vec[int, Dim8]

class _Shape:
    @overload
    def __get__(self, instance: _ArrayLike[Any, Dim1], owner: Any = None) -> tuple[int]: ...
    @overload
    def __get__(self, instance: _ArrayLike[Any, Dim2], owner: Any = None) -> tuple[int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim3], owner: Any = None
    ) -> tuple[int, int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim4], owner: Any = None
    ) -> tuple[int, int, int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim5], owner: Any = None
    ) -> tuple[int, int, int, int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim6], owner: Any = None
    ) -> tuple[int, int, int, int, int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim7], owner: Any = None
    ) -> tuple[int, int, int, int, int, int, int]: ...
    @overload
    def __get__(
        self, instance: _ArrayLike[Any, Dim8], owner: Any = None
    ) -> tuple[int, int, int, int, int, int, int, int]: ...
    @overload
    def __get__(self, instance: _ArrayLike[Any, DimAny], owner: Any = None) -> tuple[int, ...]: ...

class _ArrayLike(Generic[_El, _Dim]):
    dtype: type[_El]
    shape = _Shape()

    @overload
    def __getitem__(self: _ArrayLike[_El, Dim1], key: int, /) -> _El: ...
    @overload
    def __getitem__(self: _ArrayLike[_El, Dim2], key: _Vec2i | tuple[int, int], /) -> _El: ...
    @overload
    def __getitem__(self: _ArrayLike[_El, Dim3], key: _Vec3i | tuple[int, int, int], /) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, Dim4], key: _Vec4i | tuple[int, int, int, int], /
    ) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, Dim5], key: _Vec5i | tuple[int, int, int, int, int], /
    ) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, Dim6], key: _Vec6i | tuple[int, int, int, int, int, int], /
    ) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, Dim7], key: _Vec7i | tuple[int, int, int, int, int, int, int], /
    ) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, Dim8], key: _Vec8i | tuple[int, int, int, int, int, int, int, int], /
    ) -> _El: ...
    @overload
    def __getitem__(
        self: _ArrayLike[_El, DimAny], key: Vec[int, DimAny] | tuple[int, ...] | int, /
    ) -> _El: ...
    @overload
    def __setitem__(self: _ArrayLike[_El, Dim1], key: int, val: _El, /) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim2], key: _Vec2i | tuple[int, int], val: _El, /
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim3], key: _Vec3i | tuple[int, int, int], val: _El, /
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim4], key: _Vec4i | tuple[int, int, int, int], val: _El, /
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim5], key: _Vec5i | tuple[int, int, int, int, int], val: _El, /
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim6],
        key: _Vec6i | tuple[int, int, int, int, int, int],
        val: _El,
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim7],
        key: _Vec7i | tuple[int, int, int, int, int, int, int],
        val: _El,
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, Dim8],
        key: _Vec8i | tuple[int, int, int, int, int, int, int, int],
        val: _El,
        /,
    ) -> None: ...
    @overload
    def __setitem__(
        self: _ArrayLike[_El, DimAny], key: Vec[int, DimAny] | tuple[int, ...] | int, val: _El, /
    ) -> None: ...
