"""Generic Time Synchronization Protocols."""

from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Float, Float64, Integer, UInt32

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


class Decimate(abstract.Synchronization):
    """Decimate the samples created by another synchronization protocol.

    Supports "pre-sync" decimation or "post-sync" decimation:

    - `reference: None`: apply the synchronization policy first, then, decimate
        the synchronized pairs.
    - `reference: str`: apply decimation to the specified sensor, then apply
        synchronization.

    Args:
        sync: synchronization protocol to decimate.
        factor: decimation factor to apply.
        reference: reference sensor.
    """

    def __init__(
        self, sync: spec.Synchronization, factor: int,
        reference: str | None = None
    ) -> None:
        self.factor = factor
        self.reference = reference
        self.sync = sync

    def __call__(
        self, timestamps: dict[str, Float[np.ndarray, "_N"]]
    ) -> dict[str, Integer[np.ndarray, "M"]]:
        """Apply synchronization protocol.

        Args:
            timestamps: sensor timestamps. Each key denotes a different sensor
                name, and the value denotes the timestamps for that sensor.

        Returns:
            A dictionary, where keys correspond to each sensor, and values
                correspond to the indices which map global indices to sensor
                indices, i.e. `global[sensor, i] = sensor[sync[sensor] [i]]`.
        """
        if self.reference is not None:
            decim = {k: v for k, v in timestamps.items()}
            decim[self.reference] = timestamps[self.reference][::self.factor]
            return self.sync(decim)
        else:
            synced = self.sync(timestamps)
            return {k: v[::self.factor] for k, v in synced.items()}
