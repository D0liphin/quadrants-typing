"""`@qd.dataclass` returns one of these at runtime; the stub types the decorator as the
class instead, since no intersection type can say both. `cast(StructType, Cls).members`."""

from typing import Any, Callable

class StructType:
    members: dict[str, Any]
    """Member name to dtype, in declaration order, which is also the storage order."""
    methods: dict[str, Callable[..., Any]]
    dtype: Any
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def __getattr__(name: str) -> Any: ...
