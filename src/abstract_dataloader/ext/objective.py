"""A flexible programming model for training objectives.

!!! abstract "Programming Model"

    - An [`Objective`][.] is a callable which returns a (batched) scalar loss
      and a dictionary of metrics.
    - Objectives can be combined into a higher-order objective,
      [`MultiObjective`][.], which combines their losses and aggregates their
      metrics; specify these objectives using a [`MultiObjectiveSpec`][.].
    - [`MissingInputError`][.] is raised when a required input key or attribute
      is missing.
"""

import logging
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, cast, runtime_checkable

import numpy as np
import wadler_lindig as wl
from jaxtyping import Float, Shaped, UInt8
from typing_extensions import TypeVar

from .types import TArray

logger = logging.getLogger(__name__)

YTrue = TypeVar("YTrue", infer_variance=True)
YPred = TypeVar("YPred", infer_variance=True)


class MissingInputError(Exception):
    """Exception raised when a required input key or attribute is missing."""

    def __init__(
        self, key: str, obj: Any, kind: Literal["Key", "Attribute"] = "Key"
    ) -> None:
        super().__init__(key)
        self.key = key
        self.obj = obj
        self.kind = kind

    def __str__(self) -> str:
        """Exception messages are lazy-rendered."""
        return f"{self.kind} {self.key} not found: {wl.pformat(self.obj)}"


@dataclass(frozen=True)
class VisualizationConfig:
    """General-purpose visualization configuration.

    Objectives which make use of this configuration may ignore the provided
    values.

    Attributes:
        cols: number of columns to tile images for in-training visualizations.
        width: width of each sample when rendered.
        height: height of each sample when rendered.
        cmaps: colormaps to use, where values correspond to the name of a
            matplotlib colormap or a numpy array of enumerated RGB values.
    """

    cols: int = 8
    width: int = 512
    height: int = 256
    cmaps: Mapping[str, str | UInt8[np.ndarray, "N 3"]] = field(
        default_factory=dict
    )


@runtime_checkable
class Objective(Protocol, Generic[TArray, YTrue, YPred]):
    """Composable training objective.

    !!! note

        Metrics should use `torch.no_grad()` to make sure gradients are not
        computed for non-loss metrics!

    Type Parameters:
        - `TArray`: backend (`jax.Array`, `torch.Tensor`, etc.)
        - `YTrue`: ground truth data type.
        - `YPred`: model output data type.
    """

    @abstractmethod
    def __call__(
        self, y_true: YTrue, y_pred: YPred, train: bool = True
    ) -> tuple[Float[TArray, "batch"], dict[str, Float[TArray, "batch"]]]:
        """Training metrics implementation.

        !!! tip

            When implementing `Objective`, you can add additional arguments to
            `__call__` as needed; if using `MultiObjective`, these arguments
            can be specified via the `aux` field.

            To be a valid `Objective` type, additional arguments should be
            appended following the standard arguments (i.e., after `train`),
            and provided with default values. If these arguments are required,
            the implementation should raise an appropriate error or assertion.

        Args:
            y_true: data channels (i.e. dataloader output).
            y_pred: model outputs.
            train: Whether in training mode (i.e. skip expensive metrics).

        Returns:
            A tuple containing the loss and a dict of metric values.
        """
        ...

    def visualizations(
        self, y_true: YTrue, y_pred: YPred
    ) -> dict[str, UInt8[np.ndarray, "H W 3"]]:
        """Generate visualizations for each entry in a batch.

        This method may return an empty dict.

        !!! note

            This method should be called only from a "detached" CPU thread so
            as not to affect training throughput; the caller is responsible for
            detaching gradients and sending the data to the CPU. As such,
            implementations are free to use CPU-specific methods.

        Args:
            y_true: data channels (i.e., dataloader output).
            y_pred: model outputs.

        Returns:
            A dict, where each key is the name of a visualization, and the
                value is a stack of RGB images in HWC order, detached from
                Torch and sent to a numpy array.
        """
        return {}

    def render(
        self, y_true: YTrue, y_pred: YPred, render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Render model outputs and/or ground truth for later analysis.

        This method may return an empty dict.

        ??? question "How does this differ from `visualizations`?"

            Unlike `visualizations`, which is expected to return a single
            RGB image per batch, `render` is:

            - expected to return a unique rendered value per sample, and
            - may have arbitrary types (as long as they are a numpy arrays).

        Args:
            y_true: data channels (i.e. dataloader output).
            y_pred: model outputs.
            render_gt: whether to render ground truth data.

        Returns:
            A dict, where each key is the name of a rendered output, and the
                value is a numpy array of the rendered data (e.g., an image).
        """
        return {}


YTrueAll = TypeVar("YTrueAll", infer_variance=True)
YPredAll = TypeVar("YPredAll", infer_variance=True)


@dataclass
class MultiObjectiveSpec(Generic[YTrue, YPred, YTrueAll, YPredAll]):
    """Specification for a single objective in a multi-objective setup.

    The inputs and outputs for each objective are specified using `y_true` and
    `y_pred`:

    - `None`: The provided `y_true` and `y_pred` are passed directly to the
        objective. This means that if multiple objectives all use `None`, they
        will all receive the same data that comes from the dataloader.
    - `str`: The key indexes into a mapping which has the `y_true`/`y_pred` key,
        or an object which has a matching attribute.
    - `Sequence[str]`: Each key indexes into the layers of a nested mapping or
        object.
    - `Callable`: The callable is applied to the provided `y_true` and `y_pred`.

    !!! warning

        The user is responsible for ensuring that the `y_true` and `y_pred`
        keys or callables index the appropriate types for this objective.

    Type Parameters:
        - `YTrue`: objective ground truth data type.
        - `YHat`: objective model prediction data type.
        - `YTrueAll`: type of all ground truth data (as loaded by the
            dataloader).
        - `YHatAll`: type of all model output data (as produced by the model).

    Attributes:
        objective: The objective to use.
        weight: Weight of the objective in the overall loss.
        y_true: Key or callable to index into the ground truth data.
        y_pred: Key or callable to index into the model output data.
        aux: Auxiliary inputs indexed from ground truth data and passed to the
            objective as keyword arguments. Each key becomes a keyword argument
            name, and the value specifies how to index into ``y_true``.
    """

    objective: Objective
    weight: float = 1.0
    y_true: str | Sequence[str] | Callable[[YTrueAll], YTrue] | None = None
    y_pred: str | Sequence[str] | Callable[[YPredAll], YPred] | None = None
    aux: Mapping[
        str, str | Sequence[str] | Callable[[YTrueAll], YTrue] | None
    ] = field(default_factory=dict)

    def _index(
        self, data: Any, key: str | Sequence[str] | Callable | None
    ) -> Any:
        """Index into data using the key or callable."""

        def dereference(obj, k):
            if isinstance(obj, Mapping):
                if k not in obj:
                    raise MissingInputError(k, obj, "Key")
                return obj[k]
            else:
                if not hasattr(obj, k):
                    raise MissingInputError(k, obj, "Attribute")
                return getattr(obj, k)

        if isinstance(key, str):
            return dereference(data, key)
        elif isinstance(key, Sequence):
            for k in key:
                data = dereference(data, k)
            return data
        elif callable(key):
            return key(data)
        else:  # key is None
            return data

    def index_y_true(self, y_true: YTrueAll) -> YTrue:
        """Get indexed ground truth data.

        Args:
            y_true: All ground truth data (as loaded by the dataloader).

        Returns:
            Indexed ground truth data.
        """
        return self._index(y_true, self.y_true)

    def index_y_pred(self, y_pred: YPredAll) -> YPred:
        """Get indexed model output data.

        Args:
            y_pred: All model output data (as produced by the model).

        Returns:
            Indexed model output data.
        """
        return self._index(y_pred, self.y_pred)

    def index_aux(self, y_true: YTrueAll) -> dict[str, Any]:
        """Get indexed auxiliary inputs from ground truth data.

        Args:
            y_true: All ground truth data (as loaded by the dataloader).

        Returns:
            Dict mapping each aux key to its indexed value, ready to be
            passed as keyword arguments to the objective.
        """
        return {k: self._index(y_true, spec) for k, spec in self.aux.items()}


class MultiObjective(Objective[TArray, YTrue, YPred]):
    """Composite objective that combines multiple objectives.

    ??? example "Hydra Configuration"

        If using [Hydra](https://hydra.cc/docs/intro/) for dependency
        injection, a `MultiObjective` configuration should look like this:
        ```yaml
        objectives:
        name:
            objective:
                _target_: ...
                kwargs: ...
            weight: 1.0
            y_true: "y_true_key"
            y_pred: "y_pred_key"
        ...
        ```

    Type Parameters:
        - `YTrue`: ground truth data type.
        - `YHat`: model output data type.

    Args:
        strict: if `False`, objectives with missing keys are skipped instead of
            raising an error.
        objectives: multiple objectives, organized by name; see
            [`MultiObjectiveSpec`][^.]. Each objective can also be provided as
            a dict, in which case the key/values are passed to
            `MultiObjectiveSpec`.
    """

    def __init__(
        self, strict: bool = True, **objectives: Mapping | MultiObjectiveSpec
    ) -> None:
        if len(objectives) == 0:
            raise ValueError("At least one objective must be provided.")

        self.strict = strict
        self.objectives = {
            k: v
            if isinstance(v, MultiObjectiveSpec)
            else MultiObjectiveSpec(**v)
            for k, v in objectives.items()
        }
        # Track which objectives have already logged warnings to avoid spam
        self._warned_objectives: set[str] = set()

    def __call__(
        self, y_true: YTrue, y_pred: YPred, train: bool = True
    ) -> tuple[Float[TArray, "batch"], dict[str, Float[TArray, "batch"]]]:
        loss = 0.0
        metrics = {}
        num_successful = 0

        for k, v in self.objectives.items():
            try:
                k_loss, k_metrics = v.objective(
                    v.index_y_true(y_true),
                    v.index_y_pred(y_pred),
                    train=train,
                    **v.index_aux(y_true),
                )
                loss += k_loss * v.weight
                num_successful += 1

                for name, value in k_metrics.items():
                    metrics[f"{k}/{name}"] = value

            except MissingInputError as e:
                if self.strict:
                    raise

                # Log warning for first occurrence only
                if k not in self._warned_objectives:
                    logger.warning(
                        f"Objective '{k}' skipped due to missing input: "
                        f"{e.kind} '{e.key}' not found. "
                        f"This warning will only be shown once."
                    )
                    self._warned_objectives.add(k)

        # If strict=False and no objectives succeeded, this is an error
        if num_successful == 0:
            raise RuntimeError(
                "No valid objectives were computed. All objectives had missing "
                "inputs. Please check your data pipeline and objective "
                "specifications."
            )

        # We assure that there's at least one objective.
        loss = cast(Float[TArray, ""] | Float[TArray, "batch"], loss)
        return loss, metrics

    def visualizations(
        self, y_true: YTrue, y_pred: YPred
    ) -> dict[str, UInt8[np.ndarray, "H W 3"]]:
        images = {}
        for k, v in self.objectives.items():
            try:
                k_images = v.objective.visualizations(
                    v.index_y_true(y_true),
                    v.index_y_pred(y_pred),
                    **v.index_aux(y_true),
                )
                for name, image in k_images.items():
                    images[f"{k}/{name}"] = image

            except MissingInputError as e:
                if self.strict:
                    raise

                # Log warning for first occurrence only
                warning_key = f"{k}_visualizations"
                if warning_key not in self._warned_objectives:
                    logger.warning(
                        f"Visualizations for objective '{k}' skipped due to "
                        f"missing input: {e.kind} '{e.key}' not found. This "
                        f"warning will only be shown once."
                    )
                    self._warned_objectives.add(warning_key)

        return images

    def render(
        self, y_true: YTrue, y_pred: YPred, render_gt: bool = False
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        rendered = {}
        for k, v in self.objectives.items():
            try:
                k_rendered = v.objective.render(
                    v.index_y_true(y_true),
                    v.index_y_pred(y_pred),
                    render_gt=render_gt,
                    **v.index_aux(y_true),
                )
                for name, image in k_rendered.items():
                    rendered[f"{k}/{name}"] = image

            except MissingInputError as e:
                if self.strict:
                    raise

                # Log warning for first occurrence only
                warning_key = f"{k}_render"
                if warning_key not in self._warned_objectives:
                    logger.warning(
                        f"Rendering for objective '{k}' skipped due to missing "
                        f"input: {e.kind} '{e.key}' not found. This warning "
                        f"will only be shown once."
                    )
                    self._warned_objectives.add(warning_key)

        return rendered

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        for v in self.objectives.values():
            yield v.objective
