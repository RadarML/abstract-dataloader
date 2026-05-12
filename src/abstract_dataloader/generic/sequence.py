"""Protocol transforms for working with sequences."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Generic, TypeVar, cast

import numpy as np
from jaxtyping import Float64

from abstract_dataloader import abstract, spec

Sample = TypeVar("Sample")
SampleStack = TypeVar("SampleStack")
TMetadata = TypeVar("TMetadata", bound=spec.Metadata)


@dataclass
class Metadata(spec.Metadata):
    """Generic metadata with timestamps.

    Attributes:
        timestamps: epoch timestamps.
    """

    timestamps: Float64[np.ndarray, "N"]


class Window(
    abstract.Sensor[SampleStack, Metadata],
    Generic[SampleStack, Sample, TMetadata]
):
    """Load sensor data across a time window using a sensor transform.

    Use this class as a generic transform to give time history to any sensor:

    ```python
    sensor =  ... # implements spec.Sensor
    with_history = generic.Window(sensor, past=5, future=1, parallel=7)
    ```

    In this example, 5 past samples, the current sample, and 1 future sample
    are loaded on every index:

    ```python
    with_history[i] = [
        sensor[i], sensor[i + 1], ... sensor[i + 5], sensor[i + 6]]
                                            ^
                                # timestamp for synchronization
    ```

    Type Parameters:
        - `SampleStack`: a collated series of consecutive samples. Can simply be
            `list[Sample]`.
        - `Sample`: single observation sample type.
        - `TMetadata`: metadata type for the underlying sensor. Note that the
            `Window` wrapper doesn't actually have metadata type `TMetadata`;
            this type is just passed through from the sensor which is wrapped.

    Args:
        sensor: sensor to wrap.
        collate_fn: collate function for aggregating a list of samples; if not
            specified, the samples are simply returned as a list.
        past: number of past samples, in addition to the current sample. Set
            to `0` to disable.
        future: number of future samples, in addition to the current sample.
            Set to `0` to disable.
        crop: if `True`, crop the first `past` and last `future` samples in
            the reported metadata to ensure that all samples are fully valid.
        stride: step between exposed windows. Set to `1` for overlapping
            windows, or to the window length for non-overlapping windows.
        parallel: maximum number of samples to load in parallel; if `None`, all
            samples are loaded sequentially.
    """

    def __init__(
        self, sensor: spec.Sensor[Sample, TMetadata],
        collate_fn: Callable[[list[Sample]], SampleStack] | None = None,
        past: int = 0, future: int = 0, crop: bool = True,
        stride: int = 1, parallel: int | None = None
    ) -> None:
        self.sensor = sensor
        self.past = past
        self.future = future
        self.stride = stride
        self.parallel = parallel

        if collate_fn is None:
            collate_fn = cast(
                Callable[[list[Sample]], SampleStack], lambda x: x)
        self.collate_fn = collate_fn

        self.cropped = crop
        if crop:
            # hack for negative indexing
            _future = None if future == 0 else -future
            self.metadata = Metadata(
                timestamps=sensor.metadata.timestamps[past:_future:stride])
        else:
            self.metadata = Metadata(
                timestamps=sensor.metadata.timestamps[::stride])

    @classmethod
    def from_partial_sensor(
        cls, sensor: Callable[[str], spec.Sensor[Sample, TMetadata]],
        collate_fn: Callable[[list[Sample]], SampleStack] | None = None,
        past: int = 0, future: int = 0, crop: bool = True,
        stride: int = 1, parallel: int | None = None
    ) -> Callable[[str], "Window[SampleStack, Sample, TMetadata]"]:
        """Partially initialize from partially initialized sensor.

        Use this to create windowed sensor constructors which can be
        applied to different traces to construct a dataset. For example,
        if you have a `sensor_constructor`:

        ```python
        sensor_constructor = ...
        windowed_sensor_constructor = Window.from_partial_sensor(
            sensor_constructor, ...)

        # ... somewhere inside the dataset constructor
        sensor_instance = windowed_sensor_constructor(path_to_trace)
        ```

        Args:
            sensor: sensor *constructor* to wrap.
            collate_fn: collate function for aggregating a list of samples; if
                not specified, the samples are simply returned as a list.
            past: number of past samples, in addition to the current sample.
                Set to `0` to disable.
            future: number of future samples, in addition to the current
                sample. Set to `0` to disable.
            crop: if `True`, crop the first `past` and last `future` samples in
                the reported metadata to ensure that all samples are full
                valid.
            stride: step between exposed windows. Set to `1` for overlapping
                windows, or to the window length for non-overlapping windows.
            parallel: maximum number of samples to load in parallel; if `None`,
                all samples are loaded sequentially.
        """
        def create_wrapped_sensor(
            path: str
        ) -> Window[SampleStack, Sample, TMetadata]:
            return cls(
                sensor(path), collate_fn=collate_fn, past=past,
                future=future, crop=crop, stride=stride, parallel=parallel)

        return create_wrapped_sensor

    def __getitem__(self, index: int | np.integer) -> SampleStack:
        """Fetch measurements from this sensor, by index.

        !!! warning

            Note that `past` samples are lost at the beginning, and `future`
            samples at the end to account for the window size!

            If `crop=True`, these lost samples are taken into account by the
            `Window` wrapper; if `crop=False`, the caller must handle this.

        Args:
            index: sample index.

        Returns:
            A set of `past + 1 + future` consecutives samples. Note that there
                is a `past` offset of indices between the wrapped `Window` and
                the underlying sensor!

        Raises:
            IndexError: if `crop=False`, and the requested index is out of
                bounds (i.e., in the first `past` or last `future` samples).
        """
        center = int(index) * self.stride
        if self.cropped:
            start = center
        else:
            start = center - self.past
        window = list(range(start, start + self.past + self.future + 1))

        if window[0] < 0 or window[-1] >= len(self.sensor):
            raise IndexError(
                f"Requested invalid index {index} for uncropped "
                f"Window(past={self.past}, future={self.future}).")

        if self.parallel is not None:
            with ThreadPool(min(len(window), self.parallel)) as p:
                return self.collate_fn(p.map(self.sensor.__getitem__, window))
        else:
            return self.collate_fn(list(map(self.sensor.__getitem__, window)))

    def __repr__(self) -> str:
        """Get friendly name (passing through to the underlying sensor)."""
        return (
            f"{repr(self.sensor)} x [-{self.past}:+{self.future}:"
            f"{self.stride}]")


TRaw = TypeVar("TRaw")
TTransformed = TypeVar("TTransformed")
TCollated = TypeVar("TCollated")
TProcessed = TypeVar("TProcessed")


class SequencePipeline(
    spec.Pipeline[
        Sequence[TRaw], Sequence[TTransformed],
        Sequence[TCollated], Sequence[TProcessed]],
    Generic[TRaw, TTransformed, TCollated, TProcessed]
):
    """Transform which passes an additional sequence axis through.

    The given `Pipeline` is modified to accept `Sequence[...]` for each
    data type in its pipeline, and return a `list[...]` across the additional
    axis, thus "passing through" the axis.

    For example, suppose a sequence dataloader reads

    ```
    [
        [Raw[s=0, t=0], Raw[s=0, t=1], ... Raw[s=0, t=n]]
        [Raw[s=1, t=0], Raw[s=1, t=1], ... Raw[s=1, t=n]]
        ...
        [Raw[s=b, t=0], Raw[s=b, t=1], ... Raw[s=b, t=n]
    ]
    ```

    for sequence length `t = 0...n` and batch sample `s = 0...b`. For sequence
    length `t`, the output of the transforms will be batched with the sequence
    on the outside:

    ```
    [
        Processed[s=0...b] [t=0],
        Processed[s=0...b] [t=1],
        ...
        Processed[s=0...b] [t=n]
    ]
    ```

    Type Parameters:
        - `TRaw`, `TTransformed`, `TCollated`, `TProcessed`: see
          [`Pipeline`][abstract_dataloader.spec.].

    Args:
        pipeline: input pipeline.
    """

    def __init__(
        self, pipeline: spec.Pipeline[
            TRaw, TTransformed, TCollated, TProcessed]
    ) -> None:
        self.pipeline = pipeline

    def sample(self, data: Sequence[TRaw]) -> list[TTransformed]:
        return [self.pipeline.sample(x) for x in data]

    def collate(
        self, data: Sequence[Sequence[TTransformed]]
    ) -> list[TCollated]:
        return [self.pipeline.collate(x) for x in zip(*data)]

    def batch(self, data: Sequence[TCollated]) -> list[TProcessed]:
        return [self.pipeline.batch(x) for x in data]
