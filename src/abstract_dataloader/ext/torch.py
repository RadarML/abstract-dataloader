"""Pytorch interfaces and compatibility wrappers."""

from collections.abc import Iterable, Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import optree
import torch

from abstract_dataloader import abstract, spec

TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")


class Collate(spec.Collate[TTransformed, TCollated]):
    """Generic numpy to pytorch collation.

    Converts numpy arrays to pytorch tensors, and either stacks or concatenates
    each value.

    Type Parameters:
        - `TTransformed`: input sample type.
        - `TCollated`: output collated type.

    Args:
        mode: whether to `stack` or `concat` during collation.
    """

    def __init__(self, mode: Literal["stack", "concat"] = "concat") -> None:
        self.mode = mode

    def __call__(self, data: Sequence[TTransformed]) -> TCollated:
        if self.mode == "concat":
            return optree.tree_map(
                lambda *x: torch.concat([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore
        else:
            return optree.tree_map(
                lambda *x: torch.stack([torch.from_numpy(s) for s in x]),
                *data)  # type: ignore


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
