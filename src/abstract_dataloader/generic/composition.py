"""Transform compositions."""

from collections.abc import Sequence
from typing import Any, TypeVar, cast

import numpy as np

from abstract_dataloader import spec

TRaw = TypeVar("TRaw", bound=dict[str, Any])
TTransformed = TypeVar("TTransformed", bound=dict[str, Any])
TCollated = TypeVar("TCollated", bound=dict[str, Any])
TProcessed = TypeVar("TProcessed", bound=dict[str, Any])


class ParallelTransforms(spec.Transform[TRaw, TTransformed]):
    """Compose multiple transforms, similar to [`ParallelPipelines`][^.].

    Type Parameters:
        - `PRaw`, `PTransformed`, [`Transform`][abstract_dataloader.spec.].

    Args:
        transforms: transforms to compose. The key indicates the subkey to
            apply each transform to.
    """

    def __init__(self, **transforms: spec.Transform) -> None:
        self.transforms = transforms

    def __call__(self, data: TRaw) -> TTransformed:
        return cast(
            TTransformed,
            {k: v(data[k]) for k, v in self.transforms.items()})


class ParallelPipelines(
    spec.Pipeline[TRaw, TTransformed, TCollated, TProcessed],
):
    """Compose multiple transforms in parallel.

    For example, with transforms `{"radar": radar_tf, "lidar": lidar_tf, ...}`,
    the composed transform performs:

    ```python
    {
        "radar": radar_tf.transform(data["radar"]),
        "lidar": lidar_tf.transform(data["lidar"]),
        ...
    }
    ```

    !!! note

        This implies that the type parameters must be `dict[str, Any]`, so this
        class is parameterized by a separate set of
        `Composed(Raw|Transformed|Collated|Processed)` types with this bound.

    Type Parameters:
        - `PRaw`, `PTransformed`, `PCollated`, `PProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        transforms: transforms to compose. The key indicates the subkey to
            apply each transform to.
    """

    def __init__(self, **transforms: spec.Pipeline) -> None:
        self.transforms = transforms

    def sample(self, data: TRaw) -> TTransformed:
        return cast(
            TTransformed,
            {k: v.sample(data[k]) for k, v in self.transforms.items()})

    def collate(self, data: Sequence[TTransformed]) -> TCollated:
        return cast(TCollated, {
            k: v.collate([x[k] for x in data])
            for k, v in self.transforms.items()
        })

    def batch(self, data: TCollated) -> TProcessed:
        return cast(
            TProcessed,
            {k: v.batch(data[k]) for k, v in self.transforms.items()})


TSample = TypeVar("TSample", bound=dict[str, Any])
TSampleMeta = TypeVar("TSampleMeta", bound=dict[str, Any])


class DatasetMeta(spec.Dataset[TSampleMeta]):
    """Dataset wrapper that adds metadata to each sample.

    !!! warning

        This class assumes that the underlying data type is a dictionary, and
        merges the metadata as a new `meta` key, i.e.:
        ```python
        {"meta": meta, **dataset[index]}
        ```

    Type Parameters:
        - `TSample`, `TSampleWithMeta`: sample types before and after adding
          metadata.

    Args:
        dataset: underlying dataset.
        meta: static metadata to add to each sample.
    """

    def __init__(
        self, dataset: spec.Dataset[TSample], meta: Any
    ) -> None:
        self.dataset = dataset
        self.meta = meta

    def __getitem__(self, index: int | np.integer) -> TSampleMeta:
        """Fetch item from this dataset by global index.

        Args:
            index: sample index.

        Returns:
            Loaded sample.
        """
        return cast(TSampleMeta, {"meta": self.meta, **self.dataset[index]})

    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        return len(self.dataset)
