"""Transform compositions."""

from collections.abc import Sequence
from typing import Any, TypeVar, cast

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
