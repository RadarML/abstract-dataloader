"""Composing transforms for data processing pipelines.

!!! abstract "Programming Model"

    - Data is represented as a dictionary with string keys and arbitrary
        values which are atomic from the perspective of transform composition.
    - Transforms are created from a directed acyclic graph (DAG) of nodes,
        where each node ([`Node`][.]) is a callable which takes a set
        of inputs and produces a set of outputs.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import wadler_lindig as wl

from abstract_dataloader import spec


@dataclass
class Node:
    """Node specification for a graph-based data processing transform.

    !!! example "Example Hydra Config"

        ```yaml
        rsp:
          inputs:
            iq: radar
            aug: aug
          output: spectrum
          transform:
            _target_: grt.transforms.Spectrum
            ...
        ```

    Attributes:
        transform: callable to apply to the inputs.
        output: output data key (or output data keys for a node which returns
            multiple outputs).
        inputs: mapping of data keys to input argument names.
        optional: mapping of optional data keys to input argument names (i.e.,
            they are only passed if present).
    """

    transform: Callable
    output: str | Sequence[str]
    inputs: Mapping[str, str] = field(default_factory=dict)
    optional: Mapping[str, str] = field(default_factory=dict)

    def apply(self, data: dict[str, Any], name: str = "") -> dict[str, Any]:
        """Apply the node.

        Args:
            data: input data to process.
            name: node name (for error messages).

        Returns:
            Updated data, with any new keys added to the input data.
        """
        inputs = {k: data[v] for k, v in self.inputs.items()}
        for k, v in self.optional.items():
            if v in data:
                inputs[k] = data[v]

        output = self.transform(**inputs)

        if isinstance(self.output, str):
            data[self.output] = output
        else:  # Sequence[str]
            if not isinstance(output, Sequence):
                raise TypeError(
                    f"Node '{name}' output is expected to be a sequence due "
                    f"to output specification {self.output}: "
                    f"\n{wl.pformat(output)}\n")
            if len(self.output) != len(output):
                raise ValueError(
                    f"Node '{name}' output length mismatch: expected "
                    f"{len(self.output)} outputs ({self.output}), but got "
                    f"{len(output)} outputs:\n{wl.pformat(output)}\n")

            for o, v in zip(self.output, output):
                data[o] = v

        return data


class Transform(spec.Transform[dict[str, Any], dict[str, Any]]):
    """Compose multiple callables forming a DAG into a transform.

    !!! warning

        Since the input data specifications are not provided at initialization,
        the graph execution order (or graph validity) is not statically
        determined, and result in runtime errors if invalid.

    Args:
        outputs: output data keys to produce as a mapping of output keys to
            graph data keys. If `None`, all values are returned.
        keep_all: keep references to all intermediate values and return them
            instead of decref-ing values which are no longer needed.
        nodes: nodes in the graph, as keyword arguments where the key indicates
            a reference name for the node; any `dict` arguments are passed as
            key/value arguments to [`Node`][^.].
    """

    def __init__(
        self, outputs: Mapping[str, str] | None = None, keep_all: bool = False,
        **nodes: Node | dict[str, Any]
    ) -> None:
        self.nodes = {}
        for k, v in nodes.items():
            if not isinstance(v, Node):
                try:
                    v = Node(**v)
                except TypeError as e:
                    raise TypeError(
                        f"Node '{k}' is not a valid Node specification: "
                        f"{wl.pformat(v)}") from e
            self.nodes[k] = v

        self.outputs = outputs
        self.keep_all = keep_all

    def _err_disconnected(
        self, data: dict[str, Any], incomplete: dict[str, Node]
    ) -> ValueError:
        """Format error message for disconnected nodes."""
        remaining = {k: list(v.inputs.values()) for k, v in incomplete.items()}
        return ValueError(
            f"There are {len(incomplete)} nodes remaining, but "
            f"all remaining nodes have at least one missing input.\n"
            f"Current inputs: {list(data.keys())}\n"
            f"Remaining stage requirements: {wl.pformat(remaining)}")

    def _decref(
        self, data: dict[str, Any], incomplete: dict[str, Node]
    ) -> dict[str, Any]:
        """Decref unneeded data values."""
        if self.keep_all or self.outputs is None:
            return data
        else:
            keep = set(self.outputs.values())
            for node in self.nodes.values():
                keep |= set(node.inputs.values())
            return {k: v for k, v in data.items() if k in keep}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute the transform graph on the input data.

        Args:
            data: input data to process.

        Returns:
            Processed data.
        """
        incomplete = self.nodes.copy()

        # Guaranteed to terminate:
        # Each loop removes one node, or raises an error.
        while len(incomplete) > 0:
            for name, node in incomplete.items():
                if all(k in data for k in node.inputs.values()):
                    data = node.apply(data, name=name)
                    incomplete.pop(name)
                    data = self._decref(data, incomplete)
                    break  # break back into the while
            else:
                raise self._err_disconnected(data, incomplete)

        if self.outputs is not None:
            if self.keep_all:
                for k, v in self.outputs.items():
                    data[k] = data[v]
                return data
            else:
                return {k: data[v] for k, v in self.outputs.items()}
        else:
            return data

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        for node in self.nodes.values():
            yield node.transform

    def __repr__(self) -> str:
        """Friendly name."""
        return f"TransformGraph({', '.join(self.nodes.keys())})"
