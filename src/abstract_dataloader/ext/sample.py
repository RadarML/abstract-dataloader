"""Dataset sampling, including a low discrepancy subset sampler.

Dataset sampling is implemented using a [`SampledDataset`][.],
which transparently wraps an existing [`Dataset`][abstract_dataloader.spec.].
"""

from collections.abc import Callable, Iterable
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Int64, Integer

from abstract_dataloader import spec

TSample = TypeVar("TSample")


class SampledDataset(spec.Dataset[TSample], Generic[TSample]):
    """Dataset wrapper which only exposes a subset of values.

    The sampling `mode` can be one of:

    - `random`: Uniform random sampling, with `np.random.default_rng` and the
      supplied seed; if `seed` is a `float`, it is converted into an integer
      by multiplying by `len(dataset)` and rounding.
    - `ld`: Low discrepancy sampling; see [`sample_ld`][^.].
    - `uniform`: Uniformly spaced sampling, with `linspace(0, n, samples)`.
    - `Callable`: A callable which takes the total number of samples, and
        returns an array of indices to sample from the dataset.

    !!! info

        This `SampledDataset` is fully ADL-compliant, and acts as a passthrough
        to an ADL-compliant [`Dataset`][abstract_dataloader.spec.]: if the
        input dataset is a `Dataset[Sample]`, then the wrapped dataset is also
        a `Dataset[Sample]`.

    Type Parameters:
        `Sample`: dataset sample type.

    Args:
        dataset: underlying dataset.
        samples: target number of samples; if greater than the dataset size,
            it will be capped at the dataset size. If a `float` in `[0, 1]`,
            it is treated as a proportion of the dataset size.
        seed: sampler seed.
        mode: sampling mode.
    """

    def __init__(
        self, dataset: spec.Dataset[TSample], samples: int | float,
        seed: int | float = 0,
        mode: Literal["ld", "uniform", "random"]
            | Callable[[int], Integer[np.ndarray, "N"]] = "ld",
    ) -> None:
        self.dataset = dataset

        if isinstance(samples, float):
            samples = int(samples * len(dataset))
        samples = min(samples, len(dataset))

        if mode == "ld":
            self.subset = sample_ld(len(dataset), samples=samples, seed=seed)
        elif mode == "random":
            if isinstance(seed, float):
                seed = int(seed * len(dataset))
            self.subset = np.random.default_rng(seed).choice(
                len(dataset), size=samples, replace=True)
        elif mode == "uniform":
            self.subset = np.linspace(
                0, len(dataset) - 1, samples, dtype=np.int64)
        else:  # Callable
            self.subset = mode(len(dataset)).astype(np.int64)

    def __getitem__(self, index: int | np.integer) -> TSample:
        """Fetch item from this dataset by global index."""
        return self.dataset[self.subset[index]]

    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        return self.subset.shape[0]

    def children(self) -> Iterable[Any]:
        """Get all non-container child objects."""
        return [self.dataset]

    def __repr__(self) -> str:
        """Friendly name."""
        return f"Sampled({repr(self.dataset)}, n={len(self)})"

def sample_ld(
    total: int, samples: float | int,
    seed: float | int = 0, alpha: float | int = 2 / (np.sqrt(5) + 1),
) -> Int64[np.ndarray, "samples"]:
    """Compute deterministic low-discrepancy subset mask.

    Uses a simple `alpha * n % 1` formulation, described
    [here](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/),
    with a modification to work with integer samples:

    - For a given `total`, find the integer closest to `total * alpha` which
      is co-prime with the total. Use this as the step size.
    - Then, `1...total * alpha (mod total)` is guaranteed to visit each index
      up to `total` exactly once.

    !!! note

        The default `alpha = 1 / phi` where `phi` is the golden ratio
        `(1 + sqrt(5)) / 2` has strong [low-discrepancy sampling properties
        ](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/);
        due to the quantized nature of this function, the discrepancy may be
        larger when `total` is small.

    !!! tip

        Each of the parameters (`samples`, `seed`, `alpha`) can be
        specified as a float `[0, 1]`, and a proportion of the `total` will
        be used instead. For example, if `seed = 0.7` and `total=100`, then
        `seed = 70` will be used.

    Args:
        total: total number of samples to sample from, i.e. maximum index.
        samples: number of samples to generate. Should be less than `total`.
        seed: initial offset for the sampling sequence. Can leave this at `0`.
        alpha: step size in the sequence; the default value is the inverse
            golden ratio `2 / (np.sqrt(5) + 1)` (which is actually equivalent
            to the golden ratio `mod 1`, since `1 - phi = 1 / phi`).

    Returns:
        Array, in mixed order, of `sample` indices which form a subset of
            `0...total` with a guarantee of no repeats.
    """
    def _get_qld_step(n: int, q: int) -> int:
        """Get quantized low-discrepancy step size."""
        for i in range(min(n - q, q)):
            query =  q + (((i + 1) // 2) * (-1 if i % 2 == 0 else 1))
            if np.gcd(query, n) == 1:
                return query
        raise ValueError(
            "No GCD found. This should be impossible! Is `n` degenerate?")

    if isinstance(samples, float):
        samples = int(samples * total)
    if isinstance(seed, float):
        seed = int(seed * total)
    if isinstance(alpha, float):
        alpha = _get_qld_step(total, int(total * alpha))

    if samples > total or samples < 0:
        raise ValueError(
            f"Number of samples {samples} must be in [0, {total}].")
    if samples * alpha >= np.iinfo(np.int64).max:
        raise NotImplementedError(
            f"`samples={samples}` is too large, and will cause overflow.")

    return (np.arange(samples, dtype=np.int64) * alpha + seed) % total
