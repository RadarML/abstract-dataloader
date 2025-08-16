"""Abstract Dataloader Generic/Abstract Implementations.

The implementations here provide
[abstract](https://docs.python.org/3/library/abc.html) implementations of
commonly reusable functions such as multi-trace datasets, and glue logic for
synchronization.

- Where applicable, "polyfill" fallbacks also implement some
    methods in terms of more basic ones to allow for extending implementations
    to be more minimal, while still covering required functionality.
- In cases where fallbacks are sufficient to provide a minimal, non-crashing
    implementation of the spec, we omit the [`ABC`][abc.] base class so that
    the class is not technically abstract (though it still may be abstract,
    in the sense that it may not be meaningful to use it directly.)

Some other convenience methods are also provided which are not included in the
core spec; software using the abstract data loader should not rely on these,
and should always base their code on the [`spec`][abstract_dataloader.spec]
types.

!!! fallback

    Abstract base classes which provide default or "fallback" behavior,
    e.g. implementing some methods in terms of others, are documented with
    a `Fallback` section.

!!! note

    Classes without separate abstract implementations are also aliased to
    their original protocol definitions, so that
    [`abstract_dataloader.abstract`][abstract_dataloader.abstract]
    exposes an identical set of objects as
    [`abstract_dataloader.spec`][abstract_dataloader.spec].
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any, TypeVar, cast, overload

import numpy as np
from jaxtyping import Int64, Integer

from . import spec
from .spec import Metadata, Synchronization

__all__ = [
    "Dataset", "Metadata", "Sensor", "Synchronization", "Trace", "Pipeline",
    "Transform", "Collate"
]

TSample = TypeVar("TSample")
TMetadata = TypeVar("TMetadata", bound=spec.Metadata)
TTrace = TypeVar("TTrace", bound=spec.Trace)


class Sensor(ABC, spec.Sensor[TSample, TMetadata]):
    """Abstract Sensor Implementation.

    Type Parameters:
        - `TSample`: sample data type which this `Sensor` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.
        - `TMetadata`: metadata type associated with this sensor; must
            implement [`Metadata`][abstract_dataloader.spec.].

    Args:
        metadata: sensor metadata, including timestamp information; must
            implement [`Metadata`][abstract_dataloader.spec.].
        name: friendly name; should only be used for debugging and inspection.
    """

    def __init__(self, metadata: TMetadata, name: str = "sensor") -> None:
        self.metadata = metadata
        self.name = name

    @overload
    def stream(self, batch: None = None) -> Iterator[TSample]: ...

    @overload
    def stream(self, batch: int) -> Iterator[list[TSample]]: ...

    def stream(
        self, batch: int | None = None
    ) -> Iterator[TSample | list[TSample]]:
        """Stream values recorded by this sensor.

        Fallback:
            Manually iterate through one sample at a time, loaded using the
            provided `__getitem__` implementation.

        Args:
            batch: batch size; if `0`, returns single samples.

        Returns:
            Iterable of samples (or sequences of samples).
        """
        if batch is None:
            for i in range(len(self)):
                yield self[i]
        else:
            for i in range(len(self) // batch):
                yield [self[j] for j in range(i * batch, (i + 1) * batch)]

    @abstractmethod
    def __getitem__(
        self, index: int | np.integer
    ) -> TSample:
        """Fetch measurements from this sensor, by index.

        Args:
            index: sample index.

        Returns:
            A single sample.
        """
        ...

    def __len__(self) -> int:
        """Total number of measurements.

        Fallback:
            Return the length of the metadata timestamps.
        """
        return self.metadata.timestamps.shape[0]

    @property
    def duration(self) -> float:
        """Trace duration from the first to last sample, in seconds.

        Fallback:
            Compute using the first and last metadata timestamp.
        """
        return self.metadata.timestamps[-1] - self.metadata.timestamps[0]

    @property
    def framerate(self) -> float:
        """Framerate of this sensor, in samples/sec."""
        # `n` samples cover `n-1` periods!
        return (len(self) - 1) / self.duration

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.name}, n={len(self)})"


class Trace(spec.Trace[TSample]):
    """A trace, consisting of multiple simultaneously-recording sensors.

    Type Parameters:
        - `Sample`: sample data type which this `Sensor` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.

    Args:
        sensors: sensors which make up this trace.
        sync: synchronization protocol used to create global samples from
            asynchronous time series. If `Mapping`; the provided indices are
            used directly; if `None`, sensors are expected to already be
            synchronous (equivalent to passing `{k: np.arange(N), ...}`).
        name: friendly name; should only be used for debugging and inspection.
    """

    def __init__(
        self, sensors: Mapping[str, spec.Sensor],
        sync: (
            spec.Synchronization | Mapping[str, Integer[np.ndarray, "N"]]
            | None) = None,
        name: str = "trace"
    ) -> None:
        self.sensors = sensors
        self.name = name

        if sync is None:
            self.indices = None
        elif isinstance(sync, Mapping):
            self.indices = sync
        else:
            self.indices = sync(
                {k: v.metadata.timestamps for k, v in sensors.items()})

    @overload
    def __getitem__(self, index: str) -> Sensor: ...

    @overload
    def __getitem__(self, index: int | np.integer) -> TSample: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> TSample | spec.Sensor:
        """Get item from global index (or fetch a sensor by name).

        !!! tip

            For convenience, traces can be indexed by a `str` sensor name,
            returning that [`Sensor`][abstract_dataloader.spec.].

        Fallback:
            Reference implementation which uses the computed
            [`Synchronization`][abstract_dataloader.spec] to retrieve the
            matching indices from each sensor. The returned samples have
            sensor names as keys, and loaded data as values, matching the
            format provided as the `sensors` parameter:

            ```python
            trace[i] = {
                "sensor_a": sensor_a[synchronized_indices["sensor_a"] [i]],
                "sensor_b": sensor_a[synchronized_indices["sensor_b"] [i]],
                ...
            }
            ```

        Args:
            index: sample index, or sensor name.

        Returns:
            Loaded sample if `index` is an integer type, or the appropriate
                [`Sensor`][abstract_dataloader.spec.] if `index` is a `str`.
        """
        if isinstance(index, str):
            return self.sensors[index]

        if self.indices is None:
            return cast(TSample, {
                k: v[index] for k, v in self.sensors.items()})
        else:
            return cast(TSample, {
                k: v[self.indices[k][index].item()]
                for k, v in self.sensors.items()})

    def __len__(self) -> int:
        """Total number of sensor-tuple samples.

        Fallback:
            Returns the number of synchronized index tuples.
        """
        if self.indices is None:
            return len(list(self.sensors.values())[0])
        else:
            return list(self.indices.values())[0].shape[0]

    def __repr__(self) -> str:  # noqa: D105
        sensors = ", ".join(self.sensors.keys())
        return (
            f"{self.__class__.__name__}({self.name}, {len(self)}x[{sensors}])")

    def children(self) -> Iterable[Any]:
        """Get all child objects."""
        return self.sensors.values()


class Dataset(spec.Dataset[TSample]):
    """A dataset, consisting of multiple traces, nominally concatenated.

    Type Parameters:
        - `TSample`: sample data type which this `Sensor` returns.

    Args:
        traces: traces which make up this dataset.
    """

    def __init__(self, traces: Sequence[spec.Trace[TSample]]) -> None:
        self.traces = traces

    @cached_property
    def indices(self) -> Int64[np.ndarray, "N"]:
        """End indices of each trace, with respect to global indices."""
        lengths = np.array([len(t) for t in self.traces], dtype=np.int64)
        return np.cumsum(lengths)

    def __getitem__(self, index: int | np.integer) -> TSample:
        """Fetch item from this dataset by global index.

        !!! bug "Unsigned integer subtraction promotes to `np.float64`"

            Subtracting unsigned integers may cause numpy to promote the result
            to a floating point number. Extending implementations should be
            careful about this behavior!

            In the default implementation here, we make sure that the computed
            indices are `int64` instead of `uint64`, and always cast the input
            to an `int64`.

        Fallback:
            Supports (and assumes) random accesses; maps to datasets using
            `np.searchsorted` to search against pre-computed trace start
            indices ([`indices`][^.]), which costs on the order of 10-100us
            per call @ 100k traces.

        Args:
            index: sample index.

        Returns:
            loaded sample.

        Raises:
            IndexError: provided index is out of bounds.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} is out of bounds for dataset with length "
                f"{len(self)}.")

        if isinstance(index, np.integer):
            index = np.int64(index)

        trace = np.searchsorted(self.indices, index, side="right")
        if trace > 0:
            remainder = index - self.indices[trace - 1]
        else:
            remainder = index
        # We have to ignore type here since python's Sequence type is not
        # well defined, i.e., does not allow `np.integer` indexing even though
        # `np.integer` is interchangeable with `int`.
        return self.traces[trace][remainder]  # type: ignore

    def __len__(self) -> int:
        """Total number of samples in this dataset.

        Fallback:
            Fetch the dataset length from the trace start indices (at the cost
            of triggering index computation).
        """
        return self.indices[-1].item()

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}"
            f"({len(self.traces)} traces, n={len(self)})")

    def children(self) -> Iterable[Any]:
        """Get all child objects."""
        return self.traces


TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")


class Transform(spec.Transform[TRaw, TTransformed]):
    """Sample or batch data transform.

    !!! warning

        Transform types are not verified during initialization, and can only
        be verified using runtime type checkers when the transforms are
        applied.

    Type Parameters:
        - `TRaw`: Input data type.
        - `TTransformed`: Output data type.

    Args:
        transforms: transforms to apply sequentially; each output type
            must be the input type of the next transform.
    """

    def __init__(self, transforms: Sequence[spec.Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: TRaw) -> TTransformed:
        """Apply transforms to a batch of samples.

        Args:
            data: A `TRaw` batch.

        Returns:
            A `TTransformed` batch.
        """
        for tf in self.transforms:
            data = tf(data)
        return cast(TTransformed, data)

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        return self.transforms


class Collate(spec.Collate[TTransformed, TCollated]):
    """Data collation.

    Type Parameters:
        - `TTransformed`: Input data type.
        - `TCollated`: Output data type.
    """

    def __call__(self, data: Sequence[TTransformed]) -> TCollated:
        """Collate a set of samples.

        Args:
            data: A set of `TTransformed` samples.

        Returns:
            A `TCollated` batch.
        """
        return cast(TCollated, data)


class Pipeline(
    spec.Pipeline[TRaw, TTransformed, TCollated, TProcessed]
):
    """Dataloader transform pipeline.

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
        self._children = []
        if sample is not None:
            self.sample = sample
            self._children.append(sample)
        if collate is not None:
            self.collate = collate
            self._children.append(collate)
        if batch is not None:
            self.batch = batch
            self._children.append(batch)

    def sample(self, data: TRaw) -> TTransformed:
        """Transform single samples.

        Fallback:
            The identity transform is provided by default
            (`TTransformed = TRaw`).

        Args:
            data: A single `TRaw` data sample.

        Returns:
            A single `TTransformed` data sample.
        """
        return cast(TTransformed, data)

    def collate(self, data: Sequence[TTransformed]) -> TCollated:
        """Collate a list of data samples into a GPU-ready batch.

        Args:
            data: A sequence of `TTransformed` data samples.

        Returns:
            A `TCollated` collection of the input sequence.
        """
        return cast(TCollated, data)

    def batch(self, data: TCollated) -> TProcessed:
        """Transform data batch.

        !!! warning

            If this `Pipeline` requires GPU state in Pytorch, use
            [`ext.torch.Pipeline`][abstract_dataloader.ext.torch.Pipeline]
            instead, which implements the pipeline as a
            [`torch.nn.Module`][torch.nn.Module] instead.

        Fallback:
            The identity transform is provided by default
            (`TProcessed = TCollated`).

        Args:
            data: A `TCollated` batch of data, nominally already sent to the
                GPU.

        Returns:
            The `TProcessed` output, ready for the downstream model.
        """
        return cast(TProcessed, data)

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        return self._children
