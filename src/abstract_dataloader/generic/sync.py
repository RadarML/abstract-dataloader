"""Generic Time Synchronization Protocols."""

from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Float64, UInt32

from abstract_dataloader import abstract, spec


class Empty(spec.Synchronization):
    """Dummy synchronization which does not synchronize sensor pairs.

    No samples will be registered, and the trace can only be used as a
    collection of sensors.
    """

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        return {k: np.array([], dtype=np.uint32) for k in timestamps}


class Next(abstract.Synchronization):
    """Next sample synchronization, with respect to a reference sensor.

    See [`abstract.Synchronization`][abstract_dataloader.] for more details
    about the reference sensor and margin calculation.

    Args:
        reference: reference sensor to synchronize to.
        margin: time/index margin to apply.
    """

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        t_ref = self.get_reference(timestamps)
        return {
            k: np.searchsorted(t_sensor, t_ref).astype(np.uint32)
            for k, t_sensor in timestamps.items()}


class Nearest(abstract.Synchronization):
    """Nearest sample synchronization, with respect to a reference sensor.

    See [`abstract.Synchronization`][abstract_dataloader.] for more details
    about the reference sensor and margin calculation.

    Args:
        reference: reference sensor to synchronize to.
        margin: time/index margin to apply.
        tol: synchronization time tolerance, in seconds. Setting `tol = np.inf`
            works to disable this check altogether.
    """

    def __init__(
        self, reference: str,
        margin: Mapping[str, Sequence[int | float]]
            | Sequence[int | float] = {},
        tol: float = 0.1
    ) -> None:
        if tol < 0:
            raise ValueError(
                f"Synchronization tolerance must be positive: {tol} < 0")

        self.tol = tol
        super().__init__(reference=reference, margin=margin)

    def __call__(
        self, timestamps: dict[str, Float64[np.ndarray, "_N"]]
    ) -> dict[str, UInt32[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: input sensor timestamps.

        Returns:
            Synchronized index map.
        """
        t_ref = self.get_reference(timestamps)

        indices = {
            k: np.searchsorted(
                (t_sensor[:-1] + t_sensor[1:]) / 2, t_ref
            ).astype(np.uint32)
            for k, t_sensor in timestamps.items()}

        valid = np.all(np.array([
           np.abs(timestamps[k][i_nearest] - t_ref) < self.tol
        for k, i_nearest in indices.items()]), axis=0)

        return {k: v[valid] for k, v in indices.items()}
