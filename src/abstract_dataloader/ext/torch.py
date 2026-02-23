"""Pytorch interfaces and compatibility wrappers."""

from collections.abc import Iterable, Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import optree
import torch
from torch.nn.modules.module import _IncompatibleKeys

from abstract_dataloader import abstract, spec

TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")


class Collate(spec.Collate[TTransformed, TCollated]):
    """Generic numpy to pytorch collation.

    !!! info

        This collator uses [`optree.tree_map`][?optree.tree_map] to
        recursively traverse the input data structure. Python primitive
        containers will work out-of-the-box, while dataclasses must be
        [registered with optree](types.md#dataclass).

    | Input           | Behavior                                             |
    | --------------- | ---------------------------------------------------- |
    | `torch.Tensor`  | Either stacked or concatenated, depending on `mode`. |
    | `numpy.ndarray` | Converted to `Tensor`, then stacked/concatenated.    |
    | `int | float | bool`, `convert_scalars=True` | Converted to `Tensor`.  |
    | All other types | Passed through as a list.                            |

    Type Parameters:
        - `TTransformed`: input sample type.
        - `TCollated`: output collated type.

    Args:
        mode: whether to `stack` or `concat` during collation.
        convert_scalars: whether to convert python scalars to pytorch tensors.
    """

    def __init__(
        self, mode: Literal["stack", "concat"] = "concat",
        convert_scalars: bool = True
    ) -> None:
        self.mode = mode
        self.convert_scalars = convert_scalars

    def _convert(self, *values) -> torch.Tensor | list[Any]:
        if isinstance(values[0], np.ndarray):
            values = [torch.from_numpy(v) for v in values]

        if isinstance(values[0], torch.Tensor):
            if self.mode == "concat":
                return torch.concat(values)
            else:  # "stack"
                return torch.stack(values)
        elif self.convert_scalars and isinstance(values[0], (float, int, bool)):
            return torch.tensor(values)
        else:
            return list(values)

    def __call__(self, data: Sequence[TTransformed]) -> TCollated:
        """Apply collation.

        Args:
            data: sequence of samples to collate (i.e., list of objects).
                Must have an identical structure.

        Returns:
            Collated batch (i.e., object of lists).
        """
        return optree.tree_map(self._convert, *data)  # type: ignore


class TransformedDataset(
    torch.utils.data.Dataset[TTransformed],
    Generic[TRaw, TTransformed]
):
    """Pytorch-compatible dataset with transformation applied.

    Extends [`torch.utils.data.Dataset`][?torch.utils.data.Dataset],
    implementing a torch "map-style" dataset.

    Type Parameters:
        - `TRaw`: raw data type from the dataloader.
        - `TTransformed`: output data type from the provided transform function.

    Args:
        dataset: source dataset.
        transform: transformation to apply to each sample when loading (note
            that `Transform[TRaw, TTransformed]` is equivalent to
            `Callable[[TRaw], TTransformed]`).
    """

    def __init__(
        self, dataset: spec.Dataset[TRaw],
        transform: spec.Transform[TRaw, TTransformed]
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int | np.integer) -> TTransformed:
        """Map-style dataset indexing.

        Args:
            index: dataset index; passthrough to the underlying `Dataset`.

        Returns:
            Transformed sample.
        """
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        """Dataset length; passthrough to the underlying `Dataset`."""
        return len(self.dataset)

    def __repr__(self) -> str:
        """Friendly name."""
        return f"Transformed({repr(self.dataset)} -> {repr(self.transform)})"

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        return [self.dataset, self.transform]


class Pipeline(
    torch.nn.Module,
    abstract.Pipeline[TRaw, TTransformed, TCollated, TProcessed]
):
    """Dataloader transform pipeline.

    This pytorch-compatible pipeline extends
    [`torch.nn.Module`][torch.nn.Module]. It recursively searches its inputs
    for a `.children() -> Iterator | Iterable` method, and checks the children
    for any `nn.Module` objects, which are registered as submodules.

    Type Parameters:
        - `TRaw`: Input data format.
        - `TTransformed`: Data after the first `transform` step.
        - `TCollated`: Data after the second `collate` step.
        - `TProcessed`: Output data format.

    Args:
        sample: sample transform; if `None`, the identity transform is used
            (or the default transform, if overridden).
        collate: sample collation; if `None`, the provided default is used.
            Note that there is no fallback for collation, and
            `NotImplementedError` will be raised if none is provided.
        batch: batch collation; if `None`, the identity transform is used.
    """

    def __init__(
        self, sample: spec.Transform[TRaw, TTransformed] | None = None,
        collate: spec.Collate[TTransformed, TCollated] | None = None,
        batch: spec.Transform[TCollated, TProcessed] | None = None
    ) -> None:
        super().__init__()

        if sample is not None:
            self.sample = sample
        if collate is not None:
            self.collate = collate
        if batch is not None:
            self.batch = batch

        _modules = self._find_modules([sample, collate, batch])
        if len(_modules) > 0:
            self.submodules = torch.nn.ModuleList(_modules)

    @staticmethod
    def _find_modules(objs: Iterable) -> list[torch.nn.Module]:
        modules = []
        for obj in objs:
            if isinstance(obj, torch.nn.Module):
                modules.append(obj)
            elif hasattr(obj, "children") and callable(obj.children):
                _children = obj.children()
                if isinstance(_children, Iterable):
                    modules.extend(Pipeline._find_modules(_children))
        return modules

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(
        self, state_dict, strict: bool = True, assign: bool = True
    ) -> _IncompatibleKeys:
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
