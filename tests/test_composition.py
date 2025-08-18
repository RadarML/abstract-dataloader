"""Windowing and sequence transforms."""

import numpy as np
import pytest
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader import abstract, generic, spec  # noqa: E402


class _ConcreteMetadata(spec.Metadata):

    def __init__(self, n: int = 10):
        self.timestamps = np.arange(n, dtype=np.float64)

class _ConcreteSensor(abstract.Sensor[int, _ConcreteMetadata]):

    def __getitem__(self, index: int | np.integer) -> int:
        return int(index)


def _apply_pipeline(pipeline, raw):
    transformed = [pipeline.sample(x) for x in raw]
    collated = pipeline.collate(transformed)
    processed = pipeline.batch(collated)
    return processed


@pytest.mark.parametrize("parallel", [None, 1, 2])
def test_window(parallel):
    """Test windowing."""
    meta = _ConcreteMetadata(5)

    partialed = generic.Window.from_partial_sensor(
        lambda path: _ConcreteSensor(meta),
        past=1, future=1, parallel=parallel)
    windowed = partialed("ignored")

    assert windowed[0] == [0, 1, 2]
    assert windowed[1] == [1, 2, 3]
    assert windowed[2] == [2, 3, 4]
    assert np.all(windowed.metadata.timestamps == np.array([1, 2, 3]))


@pytest.mark.parametrize("parallel", [None, 1, 2])
def test_window_uncropped(parallel) -> None:
    """Test uncropped windowing."""
    meta = _ConcreteMetadata(5)

    partialed = generic.Window.from_partial_sensor(
        lambda path: _ConcreteSensor(meta),
        past=1, future=1, crop=False, parallel=parallel)
    windowed = partialed("ignored")

    assert windowed[1] == [0, 1, 2]
    assert windowed[2] == [1, 2, 3]
    assert windowed[3] == [2, 3, 4]
    assert np.all(windowed.metadata.timestamps == np.array([0, 1, 2, 3, 4]))

    with pytest.raises(IndexError):
        _ = windowed[0]
    with pytest.raises(IndexError):
        _ = windowed[4]


def test_transform():
    """Test sequence transforms."""
    pipeline = abstract.Pipeline(
        sample=lambda data: data * 2,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x * 5 for x in data])
    sequenced = generic.SequencePipeline(pipeline)

    raw = [[1, 4], [2, 5], [3, 6]]
    processed = _apply_pipeline(sequenced, raw)
    assert processed == [[10, 20, 30], [40, 50, 60]]


def test_parallel():
    """Test parallel transforms."""
    pipeline_a = abstract.Pipeline(
        sample=lambda data: data * 2,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x * 5 for x in data])

    pipeline_b = abstract.Pipeline(
        sample=lambda data: data + 1,  # type: ignore
        collate=lambda data: data,
        batch=lambda data: [x + 2 for x in data])

    composed = generic.ParallelPipelines(a=pipeline_a, b=pipeline_b)
    raw = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    processed = _apply_pipeline(composed, raw)
    assert processed == {"a": [10, 30], "b": [5, 7]}
