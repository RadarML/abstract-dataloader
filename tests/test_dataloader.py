"""Type checking the types used for checking."""

from typing import Sequence

import numpy as np
import pytest
from beartype.claw import beartype_package
from beartype.door import is_bearable, is_subhint

beartype_package("abstract_dataloader")

from abstract_dataloader import abstract, generic, spec  # noqa: E402


def test_spec_types():
    """Verify spec inheritance."""
    assert is_subhint(spec.Trace, spec.Dataset)
    assert is_subhint(spec.Sensor, spec.Dataset)
    assert is_subhint(spec.Collate, spec.Transform)


class _ConcreteMetadata(spec.Metadata):

    def __init__(self, n: int = 10):
        self.timestamps = np.arange(n, dtype=np.float64)

class _ConcreteSensor(abstract.Sensor[int, _ConcreteMetadata]):

    def __getitem__(self, index: int | np.integer) -> int:
        return index * 2

class _ConcreteTransform(spec.Transform):
    def __init__(self, n: int = 1):
        self.n = n

    def __call__(self, data):
        return data + self.n

class _ConcretePipeline(abstract.Pipeline[int, int, list[int], list[int]]):
    def collate(self, data: Sequence[int]) -> list[int]:
        return list(data)


def test_abstract_types():
    """Verify abstract class inheritance."""
    meta = _ConcreteMetadata(10)
    sensor = _ConcreteSensor(meta)
    trace = abstract.Trace({"sensor": sensor}, sync=generic.Next("sensor"))
    dataset = abstract.Dataset([trace, trace])
    transform = _ConcreteTransform()
    pipeline = _ConcretePipeline(sample=transform)

    assert is_bearable(meta, spec.Metadata)  # type: ignore
    assert is_bearable(sensor, spec.Sensor)  # type: ignore
    assert is_bearable(trace, spec.Trace)  # type: ignore
    assert is_bearable(dataset, spec.Dataset)  # type: ignore
    assert is_bearable(transform, spec.Transform)  # type: ignore
    assert is_bearable(pipeline, spec.Pipeline)  # type: ignore


@pytest.mark.parametrize("do_sync", [True, False, "manual"])
def test_dataset(do_sync):
    """Test dataset loading."""
    meta = _ConcreteMetadata(10)
    sensor = _ConcreteSensor(meta)

    if do_sync == "manual":
        sync = {"sensor": np.arange(10)}
    elif do_sync:
        sync = generic.Nearest("sensor")
    else:
        sync = None

    trace = abstract.Trace({"sensor": sensor}, sync=sync)
    dataset = abstract.Dataset([trace, trace])

    assert len(sensor) == 10
    assert len(trace) == 10
    assert len(dataset) == 20

    assert sensor.duration == 9.0
    assert sensor.framerate == 1.0

    assert sensor[3] == 6
    assert trace[3] == {'sensor': 6}
    assert dataset[2] == {'sensor': 4}
    assert dataset[13] == {'sensor': 6}
    assert dataset[np.uint8(13)] == {"sensor": 6}
    assert trace['sensor'] is sensor

    for i, j in enumerate(sensor.stream()):
        assert i * 2 == j

    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[20]

    acc = []
    for i in sensor.stream(batch=2):
        acc.extend(i)
    assert acc == [i * 2 for i in range(10)]


def test_transform():
    """Test transform composition."""
    transform = _ConcreteTransform()
    pipeline = _ConcretePipeline(
        sample=abstract.Transform(transforms=[transform, transform]),
        batch=lambda data: [x * 2 for x in data])
    raw = [1, 2, 3]
    transformed = [pipeline.sample(x) for x in raw]
    collated = pipeline.collate(transformed)
    processed = pipeline.batch(collated)
    assert transformed == [3, 4, 5]
    assert collated == [3, 4, 5]
    assert processed == [6, 8, 10]

    emptypipeline = _ConcretePipeline()
    raw = [1, 2]
    transformed = [emptypipeline.sample(x) for x in raw]
    collated = emptypipeline.collate(transformed)
    processed = emptypipeline.batch(collated)
    assert transformed == [1, 2]
    assert collated == [1, 2]
    assert processed == [1, 2]
