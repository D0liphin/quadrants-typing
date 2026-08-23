from typing import Annotated, Any

# To a type checker, `Runtime[TypeChecker, runtime]` is `Annotated[TypeChecker,
# runtime]`, i.e. it reads as `TypeChecker`. The runtime `annotations.py`
# instead collapses the subscription to `runtime` (the resolved value quadrants
# needs). Keeping the two views in sync is the whole point of `Runtime`.
Runtime = Annotated

# `Runtime[...]` resolves eagerly at runtime, so callers rarely need this; it
# still unwraps a bare `Annotated[T, runtime]` to `runtime`.
def _resolve_at_runtime(ann: Any) -> Any: ...
