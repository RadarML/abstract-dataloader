"""Data type system framework following the [ADL recommendations](./types.md).

!!! abstract "Programming Model"

    - Declare types as dataclasses using the provided [`dataclass`][.]
        decorator, which registers them to the global optree namespace.
    - Set each class as `Generic[TArray]`, where [`TArray`][.] is a `TypeVar`
        which is [`ArrayLike`][.], e.g., `torch.Tensor`, `jax.Array` or
        `np.ndarray`.
"""

from dataclasses import field
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from optree.dataclasses import dataclass as _optree_dataclass
from typing_extensions import dataclass_transform


@dataclass_transform(field_specifiers=(field,))
def dataclass(cls):  # noqa: D103
    """A dataclass decorator which registers into optree's global namespace."""
    return _optree_dataclass(cls, namespace='')


@runtime_checkable
class ArrayLike(Protocol):
    """Array type, e.g., `torch.Tensor | jax.Array | np.ndarray`.

    Use this type to specify arbitrary array types.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


TArray = TypeVar("TArray", bound=ArrayLike)
"""Type variable for [`ArrayLike`][^.] types."""
