"""Tests for ext.graph."""

import pytest
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader.ext import graph  # noqa: E402


def test_basic():
    """Test basic graph functionality."""
    def _node_a(a: int, increment: int = 0) -> int:
        return a + 1 + increment

    def _node_b(b: int) -> int:
        return b + 1

    def _node_ab(a: int, b: int) -> int:
        return a + b

    g = {
        "node_a": graph.Node(
            transform=_node_a,
            output="a1",
            inputs={"a": "a_in"},
            optional={"increment": "inc"}
        ),
        "node_b": {
            "transform": _node_b,
            "output": "b1",
            "inputs": {"b": "b_in"},
        },
        "node_ab": graph.Node(
            transform=_node_ab,
            output="ab_out",
            inputs={"a": "a1", "b": "b1"},
        )
    }

    tf_base = graph.Transform(outputs={"out": "ab_out"}, keep_all=False, **g)
    transformed = tf_base({"a_in": 1, "b_in": 2})
    assert transformed == {"out": 5}

    transformed = tf_base({"a_in": 1, "b_in": 2, "inc": 10})
    assert transformed == {"out": 15}

    tf_keepall = graph.Transform(outputs={"out": "ab_out"}, keep_all=True, **g)
    transformed2 = tf_keepall({"a_in": 1, "b_in": 2})
    assert transformed2 == {
        "a_in": 1, "b_in": 2, "a1": 2, "b1": 3, "ab_out": 5, "out": 5}

    tf_no_outputs = graph.Transform(keep_all=False, **g)
    transformed3 = tf_no_outputs({"a_in": 1, "b_in": 2})
    assert transformed3 == {
        "a_in": 1, "b_in": 2, "a1": 2, "b1": 3, "ab_out": 5}


def test_multi_output():
    """Test node with multiple outputs."""
    def _node_multi(a: int) -> tuple[int, int]:
        return a + 1, a + 2

    def _node_single(a: int) -> int:
        return a

    tf = graph.Transform(
        node=graph.Node(
            transform=_node_multi, output=["o1", "o2"], inputs={"a": "a"}))
    transformed = tf({"a": 1})
    assert transformed == {"a": 1, "o1": 2, "o2": 3}

    with pytest.raises(ValueError, match="There are 1 nodes remaining"):
        tf({"x": 0})

    with pytest.raises(ValueError, match="Node 'node' output length mismatch"):
        tf = graph.Transform(
            node=graph.Node(
                transform=_node_multi, output=["o1", "o2", "o3"],
                inputs={"a": "a"}))
        transformed = tf({"a": 1})

    with pytest.raises(
            TypeError,
            match="Node 'node' output is expected to be a sequence"
    ):
        tf = graph.Transform(
            node=graph.Node(
                transform=_node_single,
                output=["o1", "o2"], inputs={"a": "a"}))
        transformed = tf({"a": 1})

