"""Synchronization primitives."""

import numpy as np
import pytest
from beartype.claw import beartype_package
from beartype.door import is_bearable

beartype_package("abstract_dataloader")

from abstract_dataloader import generic, spec  # noqa: E402


def test_sync_types():
    """Test synchronization protocols."""
    assert is_bearable(generic.Empty(), spec.Synchronization)  # type: ignore
    assert is_bearable(generic.Next("dummy"), spec.Synchronization)  # type: ignore
    assert is_bearable(generic.Nearest("dummy"), spec.Synchronization)  # type: ignore


def _make_timestamps():
    return {
        "sensor1": np.arange(8, dtype=np.float64),
        "sensor2": np.arange(7, dtype=np.float64) + 0.5,
        "sensor3": np.arange(6, dtype=np.float64) + 1.0
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
    sync_ref = generic.Next("sensor1")
    ts = _make_timestamps()

    sync = generic.Next("sensor1", margin=(1, 1))
    indices = sync(ts)
    ts_ref = {k: v[1:-1] for k, v in ts.items()}
    indices_ref = sync_ref(ts_ref)
    assert all(np.array_equal(indices[k], indices_ref[k] + 1) for k in ts)

    sync2 = generic.Next("sensor1", margin=(0.1, 1.1))
    indices = sync2(ts)
    ts2_ref = {k: v[1:-2] for k, v in ts.items()}
    indices2_ref = sync_ref(ts2_ref)
    assert all(np.array_equal(indices[k], indices2_ref[k] + 1) for k in ts)

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

    sync_ref = generic.Nearest("sensor1")

    sync = generic.Nearest("sensor1", margin=(1, 1))
    indices = sync(ts)
    ts_ref = {k: v[1:-1] for k, v in ts.items()}
    indices_ref = sync_ref(ts_ref)
    assert all(np.array_equal(indices[k], indices_ref[k] + 1) for k in ts)

    sync2 = generic.Nearest("sensor1", margin=(0.1, 1.1))
    indices = sync2(ts)
    ts2_ref = {k: v[1:-2] for k, v in ts.items()}
    indices2_ref = sync_ref(ts2_ref)
    assert all(np.array_equal(indices[k], indices2_ref[k] + 1) for k in ts)


def test_decimate():
    """generic.Decimate."""
    ts = _make_timestamps()

    # Test post-sync decimation (reference=None)
    base = generic.Next("sensor1")
    post_sync = generic.Decimate(base, factor=2)
    base_indices = base(ts)
    decimated_indices = post_sync(ts)

    # Check that decimated indices are every other index from base
    for sensor in ts.keys():
        expected = base_indices[sensor][::2]
        assert np.array_equal(decimated_indices[sensor], expected)

    # Test pre-sync decimation (reference specified)
    pre_sync = generic.Decimate(base, factor=2, reference="sensor1")
    pre_decimated = pre_sync(ts)

    # Should have fewer samples since we decimate the reference first
    assert len(pre_decimated["sensor1"]) <= len(base_indices["sensor1"])

    # Test with different factor
    decimate_factor3 = generic.Decimate(base, factor=3)
    decimated_factor3 = decimate_factor3(ts)
    for sensor in ts.keys():
        expected = base_indices[sensor][::3]
        assert np.array_equal(decimated_factor3[sensor], expected)


def test_arg_checking():
    """Test argument checking for `abstract.Synchronization`."""
    with pytest.raises(TypeError):
        generic.Next(reference="sensor1", margin={"sensor1": (1,)})

    with pytest.raises(TypeError):
        generic.Next(reference="sensor1", margin={"sensor1": (1, 2, 3)})

    with pytest.raises(TypeError):
        generic.Next(reference="sensor1", margin=[1, 2, 3])
