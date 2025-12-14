"""Tests for ext.objective."""

import numpy as np
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader.ext import objective  # noqa: E402


class MockObjective:
    """Simple objective for testing."""

    def __call__(self, y_true, y_pred, train=True):
        # Return simple loss and metrics
        loss = np.array([1.0])
        metrics = {"accuracy": np.array([0.9])}
        return loss, metrics

    def visualizations(self, y_true, y_pred):
        return {"plot": np.ones((64, 64, 3), dtype=np.uint8)}

    def render(self, y_true, y_pred, render_gt=False):
        return {"rendered": np.ones((1, 32, 32))}


def test_visualization_config():
    """Test VisualizationConfig dataclass."""
    # Default values
    config = objective.VisualizationConfig()
    assert config.cols == 8
    assert config.width == 512
    assert config.height == 256
    assert config.cmaps == {}

    # Custom values
    custom_config = objective.VisualizationConfig(
        cols=4, width=256, height=128, cmaps={"depth": "viridis"}
    )
    assert custom_config.cols == 4
    assert custom_config.cmaps == {"depth": "viridis"}


def test_objective_protocol():
    """Test custom objective following the protocol."""
    obj = MockObjective()
    y_true = {"data": np.array([1, 2, 3])}
    y_pred = {"output": np.array([1.1, 2.1, 2.9])}

    # Test required __call__ method
    loss, metrics = obj(y_true, y_pred, train=True)
    assert isinstance(loss, np.ndarray)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

    # Test optional methods
    vis = obj.visualizations(y_true, y_pred)
    assert isinstance(vis, dict)

    render = obj.render(y_true, y_pred, render_gt=True)
    assert isinstance(render, dict)


def test_multi_objective_spec_indexing():
    """Test MultiObjectiveSpec indexing functionality."""
    obj = MockObjective()
    spec = objective.MultiObjectiveSpec(objective=obj, weight=2.0)

    # Test None indexing (pass-through)
    data = {"key": "value"}
    assert spec.index_y_true(data) is data
    assert spec.index_y_pred(data) is data

    # Test string indexing
    spec_str = objective.MultiObjectiveSpec(
        objective=obj, y_true="ground_truth", y_pred="predictions"
    )
    data_dict = {"ground_truth": [1, 2, 3], "predictions": [1.1, 2.1, 2.9]}
    assert spec_str.index_y_true(data_dict) == [1, 2, 3]
    assert spec_str.index_y_pred(data_dict) == [1.1, 2.1, 2.9]

    # Test callable indexing
    spec_callable = objective.MultiObjectiveSpec(
        objective=obj,
        y_true=lambda x: x["data"][:2],  # type: ignore
        y_pred=lambda x: x["output"] * 2  # type: ignore
    )
    data_for_callable = {"data": [1, 2, 3, 4], "output": 5}
    assert spec_callable.index_y_true(data_for_callable) == [1, 2]
    assert spec_callable.index_y_pred(data_for_callable) == 10


def test_multi_objective_spec_errors():
    """Test MultiObjectiveSpec error handling."""
    obj = MockObjective()
    spec = objective.MultiObjectiveSpec(objective=obj, y_true="missing_key")

    # Should raise KeyError for missing keys
    data = {"existing_key": "value"}
    try:
        spec.index_y_true(data)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_multi_objective_basic():
    """Test basic MultiObjective functionality."""
    obj1 = MockObjective()
    obj2 = MockObjective()

    multi_obj = objective.MultiObjective(
        task1={"objective": obj1, "weight": 1.0},
        task2={"objective": obj2, "weight": 0.5}
    )

    y_true = {"data": np.array([1, 2, 3])}
    y_pred = {"output": np.array([1.1, 2.1, 2.9])}

    # Test loss combination (1.0 * 1.0 + 0.5 * 1.0 = 1.5)
    loss, metrics = multi_obj(y_true, y_pred, train=True)
    assert isinstance(loss, (int, float, np.ndarray))

    # Test namespaced metrics
    assert "task1/accuracy" in metrics
    assert "task2/accuracy" in metrics

    # Test visualization delegation
    vis = multi_obj.visualizations(y_true, y_pred)
    assert "task1/plot" in vis
    assert "task2/plot" in vis

    # Test render delegation
    rendered = multi_obj.render(y_true, y_pred)
    assert "task1/rendered" in rendered
    assert "task2/rendered" in rendered

    # Test children method
    children = list(multi_obj.children())
    assert len(children) == 2
    assert obj1 in children
    assert obj2 in children


def test_multi_objective_error_handling():
    """Test MultiObjective error handling in strict vs non-strict modes."""
    obj = MockObjective()

    # Strict mode (default) - should raise errors
    strict_multi = objective.MultiObjective(
        task={"objective": obj, "y_true": "missing_key"}
    )

    try:
        strict_multi({"existing": "data"}, {"pred": "data"}, train=True)
        assert False, "Should have raised KeyError in strict mode"
    except KeyError:
        pass

    # Non-strict mode - should skip missing objectives
    non_strict_multi = objective.MultiObjective(
        strict=False,
        good_task={"objective": obj, "y_true": None},
        bad_task={"objective": obj, "y_true": "missing_key"}
    )

    # Should work without error, only processing good_task
    loss, metrics = non_strict_multi({"data": "value"}, {"pred": "data"})
    assert isinstance(loss, (int, float, np.ndarray))
    assert "good_task/accuracy" in metrics
    assert "bad_task/accuracy" not in metrics


def test_multi_objective_empty():
    """Test MultiObjective validation."""
    # Should raise ValueError when no objectives provided
    try:
        objective.MultiObjective()
        assert False, "Should have raised ValueError for empty objectives"
    except ValueError:
        pass
