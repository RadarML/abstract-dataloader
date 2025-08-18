"""Synchronization primitives."""

import numpy as np
import pytest
from beartype.claw import beartype_package
from beartype.door import is_bearable

beartype_package("abstract_dataloader")

from abstract_dataloader import generic, spec  # noqa: E402


def test_sync_types():
    """Test synchronization protocols."""
    assert is_bearable(generic.Empty(), spec.Synchronization)
    assert is_bearable(generic.Next("dummy"), spec.Synchronization)
    assert is_bearable(generic.Nearest("dummy"), spec.Synchronization)


def _make_timestamps():
    return {
        "sensor1": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        "sensor2": np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        "sensor3": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    }


def test_empty():
    """generic.Empty."""
    ts = _make_timestamps()
    empty = generic.Empty()(ts)
    for k in ts:
        assert empty[k].shape == (0,)


@pytest.mark.parametrize("reference", ["sensor1", "sensor2", "sensor3"])
def test_next(reference):
    """generic.Next."""
    ts = _make_timestamps()
    sync = generic.Next(reference)
    indices = sync(ts)

    assert set(indices.keys()) == set(ts.keys())

    lengths = [len(v) for v in indices.values()]
    assert all(x == lengths[0] for x in lengths)

    for v in indices.values():
        assert np.all(np.diff(v) >= 0)


def test_next_margin():
    """generic.Next, with margin."""
    ts = _make_timestamps()

    sync = generic.Next("sensor1", margin=(1, 1))
    indices = sync(ts)
    assert indices["sensor1"].shape[0] == 1
    assert np.allclose(indices["sensor1"], [2])

    sync2 = generic.Next("sensor1", margin=(0.1, 1.1))
    indices = sync2(ts)
    assert indices["sensor1"].shape[0] == 1
    assert np.allclose(indices["sensor1"], [2])


def test_next_missing():
    """generic.Next, missing reference."""
    sync = generic.Next(reference="sensorX")
    with pytest.raises(KeyError):
        sync(_make_timestamps())


@pytest.mark.parametrize("tol", [0.05, 0.55, np.inf])
def test_nearest(tol):
    """generic.Nearest."""
    ts = _make_timestamps()
    sync = generic.Nearest(reference="sensor3", tol=tol)
    indices = sync(ts)

    assert set(indices.keys()) == set(ts.keys())
    lengths = [len(v) for v in indices.values()]
    assert all(x == lengths[0] for x in lengths)

    if tol < np.inf:
        t_ref = ts["sensor3"]
        for k in indices:
            selected_times = ts[k][indices[k]]
            diff = np.abs(selected_times - t_ref[:len(selected_times)])
            assert np.all(diff < tol)


def test_nearest_missing():
    """generic.Nearest; missing key."""
    sync = generic.Nearest(reference="sensorX", tol=0.05)
    with pytest.raises(KeyError):
        sync(_make_timestamps())


def test_nearest_bounds():
    """generic.Nearest; invalid tol."""
    with pytest.raises(ValueError):
        sync = generic.Nearest(reference="sensorX", tol=-0.5)  # noqa


def test_nearest_margin():
    """generic.Nearest, with margin."""
    ts = _make_timestamps()

    sync = generic.Nearest("sensor1", tol=1.0, margin=(1, 1))
    indices = sync(ts)
    assert indices["sensor1"].shape[0] == 4
    assert np.allclose(indices["sensor1"], [1, 2, 3, 4])

    sync2 = generic.Nearest("sensor1", tol=1.0, margin=(0.1, 1.1))
    indices = sync2(ts)
    assert indices["sensor1"].shape[0] == 3
    assert np.allclose(indices["sensor1"], [1, 2, 3])
