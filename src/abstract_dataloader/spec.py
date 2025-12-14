"""Abstract Dataloader Specifications.

The implementations here provide "duck type"
[protocol](https://peps.python.org/pep-0544/) definitions of key data loading
primitives. In order to implement the specification, users simply need to
"fill in" the methods described here for the types which they wish to
implement.

!!! type-parameters "Type Parameters"

    ADL specification protocol types are defined as
    [generics](https://typing.python.org/en/latest/reference/generics.html),
    which are parameterized by other types. These type parameters are
    documented by a **Type Parameters** section where applicable.
"""

from collections.abc import Iterator, Sequence
from typing import Generic, Protocol, overload, runtime_checkable

import numpy as np
from jaxtyping import Float, Integer
from typing_extensions import TypeVar

__all__ = [
    "Dataset", "Metadata", "Sensor", "Synchronization", "Trace", "Pipeline",
    "Transform", "Collate"
]


@runtime_checkable
class Metadata(Protocol):
    """Sensor metadata.

    All sensor metadata is expected to be held in memory during training, so
    great effort should be taken to minimize its memory usage. Any additional
    information which is not strictly necessary for book-keeping, or which
    takes more than negligible space, should be loaded as data instead.

    !!! note

        This can be a `@dataclass`, [`typing.NamedTuple`][typing.NamedTuple],
        or a fully custom type - it just has to expose a `timestamps`
        attribute.

    Attributes:
        timestamps: measurement timestamps, in seconds. Nominally in epoch
            time; must be consistent within each trace (but not necessarily
            across traces). Suggested type: `float64,` which gives precision of
            <1us.
    """

    timestamps: Float[np.ndarray, "N"]


TSample = TypeVar("TSample", covariant=True)
TMetadata = TypeVar("TMetadata", bound=Metadata)


@runtime_checkable
class Sensor(Protocol, Generic[TSample, TMetadata]):
    """A sensor, consisting of a synchronous time-series of measurements.

    This protocol is parameterized by generic `TSample` and `TMetadata` types,
    which can encode the expected data type of this sensor. For example:

    ```python
    class Point2D(TypedDict):
        x: float
        y: float

    def point_transform(point_sensor: Sensor[Point2D, Any]) -> T:
        ...
    ```

    This encodes an argument, `point_sensor`, which expected to be a sensor
    that reads data with type `Point2D`, but does not specify a metadata type.

    Type Parameters:
        - `TSample`: sample data type which this `Sensor` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.
        - `TMetadata`: metadata type associated with this sensor; must
            implement [`Metadata`][^.].

    Attributes:
        metadata: sensor metadata, including timestamp information.
    """

    metadata: TMetadata

    def stream(self) -> Iterator[TSample]:
        """Stream values recorded by this sensor.

        Returns:
            An iterator yielding successive samples.
        """
        ...

    def __getitem__(self, index: int | np.integer) -> TSample:
        """Fetch measurements from this sensor, by index.

        Args:
            index: sample index, in the sensor scope.

        Returns:
            Loaded sample.
        """
        ...

    def __len__(self) -> int:
        """Total number of measurements."""
        ...


@runtime_checkable
class Synchronization(Protocol):
    """Synchronization protocol for asynchronous time-series.

    Defines a rule for creating matching sensor index tuples which correspond
    to some kind of global index.

    !!! info "Generic Implementations"

        The following generic implementations are included with
        [`abstract_dataloader.generic`][abstract_dataloader.generic]:

        - [`Empty`][abstract_dataloader.generic.Empty]: a no-op for
            intializing a trace without any synchronization (i.e., just as a
            container of sensors).
        - [`Nearest`][abstract_dataloader.generic.Nearest]: find the nearest
            measurement for each sensor relative to the reference sensor's
            measurements.
        - [`Next`][abstract_dataloader.generic.Next]: find the next
            measurement for each sensor relative to the reference sensor's
            measurements.
        - [`Decimate`][abstract_dataloader.generic.Decimate]: a higher order
            policy which decimates the frames selected by another protocol.
    """

    def __call__(
        self, timestamps: dict[str, Float[np.ndarray, "_N"]]
    ) -> dict[str, Integer[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: sensor timestamps. Each key denotes a different sensor
                name, and the value denotes the timestamps for that sensor.

        Returns:
            A dictionary, where keys correspond to each sensor, and values
                correspond to the indices which map global indices to sensor
                indices, i.e. `global[sensor, i] = sensor[sync[sensor] [i]]`.
        """
        ...


@runtime_checkable
class Trace(Protocol, Generic[TSample]):
    """A sensor trace, consisting of multiple simultaneously-recording sensors.

    This protocol is parameterized by a generic `Sample` type, which can encode
    the expected data type of this trace.

    !!! question "Why not `Sequence`?"

        While collections of simultaneously-recorded sensor data are commonly
        referred to as "sequences," the `Sequence` name conflicts with the
        python standard-library
        [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)
        type. Since the abstract dataloader heavily references these
        [abstract types][collections.abc] (and you should too!), we use "Trace"
        to avoid conflicts.

    Type Parameters:
        - `Sample`: sample data type which this `Trace` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.
    """

    @overload
    def __getitem__(self, index: str) -> Sensor: ...

    @overload
    def __getitem__(self, index: int | np.integer) -> TSample: ...

    def __getitem__(self, index: int | np.integer | str) -> TSample | Sensor:
        """Get item from global index (or fetch a sensor by name).

        !!! info

            For the user's convenience, traces can be indexed by a `str` sensor
            name, returning that [`Sensor`][^^.]. While we are generally wary
            of requiring "quality of life" features, we include this since a
            simple `isinstance(index, str)` check suffices to implement this
            feature.

        Args:
            index: sample index, or sensor name.

        Returns:
            Loaded sample if `index` is an integer type, or the appropriate
                [`Sensor`][^^.] if `index` is a `str`.
        """
        ...

    def __len__(self) -> int:
        """Total number of sensor-tuple samples."""
        ...


@runtime_checkable
class Dataset(Protocol, Generic[TSample]):
    """A dataset, consisting of multiple traces concatenated together.

    !!! note "[`Trace`][^.] subtypes [`Dataset`][^.]"

        Due to the type signatures, a [`Trace`][^.] is actually a subtype of
        [`Dataset`][^.]. This means that a dataset which implements a
        collection of traces can also take a collection of datasets!

    Type Parameters:
        - `TSample`: sample data type which this `Dataset` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.
    """

    def __getitem__(self, index: int | np.integer) -> TSample:
        """Fetch item from this dataset by global index.

        Args:
            index: sample index.

        Returns:
            Loaded sample.
        """
        ...

    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        ...


TRaw = TypeVar("TRaw", infer_variance=True)
TTransformed = TypeVar("TTransformed", infer_variance=True)
TCollated = TypeVar("TCollated", infer_variance=True)
TProcessed = TypeVar("TProcessed", infer_variance=True)


@runtime_checkable
class Transform(Protocol, Generic[TRaw, TTransformed]):
    """Sample or batch-wise transform.

    !!! note

        This protocol is a suggestively-named equivalent to
        `Callable[[TRaw], TTransformed]` or `Callable[[Any], Any]`.

    Type Parameters:
        - `TRaw`: Input data type.
        - `TTransformed`: Output data type.
    """

    def __call__(self, data: TRaw) -> TTransformed:
        """Apply transform to a single sample.

        Args:
            data: A single `TRaw` data sample.

        Returns:
            A single `TTransformed` data sample.
        """
        ...


@runtime_checkable
class Collate(Protocol, Generic[TTransformed, TCollated]):
    """Data collation.

    !!! note

        This protocol is a equivalent to
        `Callable[[Sequence[TTransformed]], TCollated]`. `Collate` can also
        be viewed as a special case of `Transform`, where the input type
        `TRaw` must be a `Sequence[...]`.

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
        ...


@runtime_checkable
class Pipeline(
    Protocol, Generic[TRaw, TTransformed, TCollated, TProcessed]
):
    """Dataloader transform pipeline.

    This protocol is parameterized by four type variables which encode the
    different data formats at each stage in the pipeline. This forms a
    `Raw -> Transformed -> Collated -> Processed` pipeline with three
    transforms:

    | Method | Type | Description |
    |--------|------|-------------|
    | [`sample`][.] | `Raw -> Transformed` | A sample to sample transform; \
        can be sequentially assembled from one or more [`Transform`][^.]s. |
    | [`collate`][.] | `List[Transformed] -> Collated` | A list-of-samples to \
        batch transform. Can use exactly one [`Collate`][^.]. |
    | [`batch`][.] | `Collated -> Processed` | A batch to batch transform; \
        can be sequentially assembled from one or more [`Transform`][^.]s. |

    Type Parameters:
        - `TRaw`: Input data format.
        - `TTransformed`: Data after the first `transform` step.
        - `TCollated`: Data after the second `collate` step.
        - `TProcessed`: Output data format.
    """

    def sample(self, data: TRaw) -> TTransformed:
        """Transform single samples, nominally on the CPU side.

        Args:
            data: A single `TRaw` data sample.

        Returns:
            A single `TTransformed` data sample.
        """
        ...

    def collate(self, data: Sequence[TTransformed]) -> TCollated:
        """Collate a list of data samples into a GPU-ready batch.

        This method is analogous to the `collate_fn` of a
        [pytorch dataloader](https://pytorch.org/docs/stable/data.html),
        and is responsible for aggregating individual samples into a batch on
        the CPU side (but not transferring to the GPU).

        !!! info "Pytorch Implementation"

            We provide a generic implementation of this method in
            [`ext.torch.Collate`][abstract_dataloader.].

        Args:
            data: A sequence of `TTransformed` data samples.

        Returns:
            A `TCollated` collection of the input sequence.
        """
        ...

    def batch(self, data: TCollated) -> TProcessed:
        """Transform data batch, nominally on the GPU side.

        !!! info "Implementation as `torch.nn.Module`"

            If this `Pipeline` requires GPU state, it may be helpful to
            implement it as a `torch.nn.Module`. In this case, `batch`
            should redirect to `__call__`, which in turn redirects to
            [`nn.Module.forward`][torch.] in order to handle any registered
            pytorch hooks.

        Args:
            data: A `TCollated` batch of data, nominally already sent to the
                GPU.

        Returns:
            The `TProcessed` output, ready for the downstream model.
        """
        ...
