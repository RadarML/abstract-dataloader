"""Dataloader / Pytorch Lightning Datamodule bridge.

!!! warning

    [Pytorch lightning](https://lightning.ai/docs/pytorch/stable/) must be
    installed to use this module. This is not included in any extras; you will
    need to `pip install lightning` or add it to your dependencies.

The provided data module is based on the following assumptions:

- All splits use the same transform [`Pipeline`][abstract_dataloader.spec.],
    but each have a different [`Dataset`][abstract_dataloader.spec.Dataset].
    This means that if any data augmentations are applied by the transform, the
    `Dataset` should pass some `meta` information (i.e., whether in training
    mode) as part of the data.
- In-training visualizations are always rendered from the same set of a
    relatively small number of samples taken from the validation set.
- The same dataloader settings should be applied to all splits.

!!! info

    Only sample-to-sample (`.transform`) and sample-to-batch (`.collate`)
    transforms are applied in the dataloader; the training loop is
    responsible for applying batch-to-batch (`.forward`) transforms.
"""

from functools import cache, cached_property, partial
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeVar

import lightning as L  # noqa: N812
import numpy as np
from torch.utils.data import DataLoader

from abstract_dataloader import spec
from abstract_dataloader.ext.torch import TransformedDataset

from .sample import SampledDataset

Sample = TypeVar("Sample")
Raw = TypeVar("Raw", contravariant=True, bound=dict[str, Any])
Transformed = TypeVar("Transformed", bound=dict[str, Any])
Collated = TypeVar("Collated", bound=dict[str, Any])
Processed = TypeVar("Processed", covariant=True, bound=dict[str, Any])


class ADLDataModule(
    L.LightningDataModule, Generic[Raw, Transformed, Collated, Processed]
):
    """Pytorch dataloader wrapper for ADL-compliant datasets.

    !!! info

        Train/val/test splits are not all required to be present; if any are
        not present, the corresponding `.{split}_dataloader()` will raise an
        error if called. Arbitrary split names are also allowed, though
        `train`, `val`, and `test` are expected for the
        `ADLDataModule.{train|val|test}_dataloader()` methods
        [expected by pytorch lightning](
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

    !!! note

        The underlying (transformed) dataset is cached (i.e. the same
        dataset object will be used on each call), but the dataloader
        container is not.

    Type Parameters:
        - `Raw`: raw data loaded from the dataset.
        - `Transformed`: data following CPU-side transform.
        - `Collated`: data format after collation; should be in pytorch tensors.
        - `Processed`: data after GPU-side transform.

    Args:
        dataset: datasets or dataset constructors for each split.
        transforms: data transforms to apply.
        batch_size: dataloader batch size.
        samples: number of validation-set samples to prefetch for
            visualizations (or a list of indices to use). Note that these
            samples are always held in memory! Set `samples=0` to disable.
        num_workers: number of worker processes during data loading and
            CPU-side processing; use `num_workers=0` to run in the main thread.
        prefetch_factor: number of batches to fetch per worker. Must be `None`
            when `num_workers=0`.
        subsample: Sample only a (low-discrepancy) subset of samples on each
            split specified here instead of using all samples.

    Attributes:
        transforms: data transforms which should be applied to the data; in
            particular, a `.forward()` GPU batch-to-batch stage which is
            expected to be handled by downstream model code.
    """

    def __init__(
        self, dataset: Mapping[
            str, Callable[[], spec.Dataset[Raw]] | spec.Dataset[Raw]],
        transforms: spec.Pipeline[Raw, Transformed, Collated, Processed],
        batch_size: int = 32, samples: int | Sequence[int] = 0,
        num_workers: int = 32, prefetch_factor: int | None = None,
        subsample: Mapping[str, int | float | None] = {}
    ) -> None:
        super().__init__()

        self.transforms = transforms
        self._samples = samples
        self._subsample = subsample

        self._dataset = dataset
        self._dataloader_args = {
            "batch_size": batch_size, "num_workers": num_workers,
            "prefetch_factor": prefetch_factor, "pin_memory": True,
            "collate_fn": transforms.collate
        }

    @classmethod
    def from_traces(
        cls, dataset: Callable[[Sequence[str]], spec.Dataset[Raw]],
        traces: Mapping[str, Sequence[str]],
        transforms: spec.Pipeline[Raw, Transformed, Collated, Processed],
        **kwargs: dict[str, Any]
    ) -> "ADLDataModule[Raw, Transformed, Collated, Processed]":
        """Create from a dataset constructor.

        Args:
            dataset: dataset constructor which takes a list of trace names
                and returns a dataset object.
            traces: mapping of split names to trace names; the dataset
                constructor will be called with the trace names for each split.
            transforms: data transforms to apply.
            kwargs: see the class constructor.
        """
        return cls(
            dataset={k: partial(dataset, v) for k, v in traces.items()},
            transforms=transforms,
            **kwargs)  # type: ignore

    @cached_property
    def samples(self) -> Collated | None:
        """Validation samples for rendering samples.

        If a simple `samples: int` is specified, these samples are taken
        uniformly `len(val) // samples` apart with padding on either side.

        !!! warning

            While this property is cached, accessing this property the first
            time triggers a full load of the dataset validation split!

        Returns:
            Pre-loaded validation samples, nominally for generating
                visualizations. If `samples=0` was specified, or no validation
                split is provided, then no samples are returned.
        """
        if self._samples != 0:
            try:
                val = self.dataset("val")

                if isinstance(self._samples, int):
                    m = len(val) // self._samples // 2
                    indices = np.linspace(
                        m, len(val) - m, self._samples, dtype=np.int64)
                else:
                    indices = self._samples

                sampled = [val[i] for i in indices]
                return self.transforms.collate(sampled)
            except KeyError:
                return None
        return None

    @cache
    def dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> TransformedDataset[Raw, Transformed]:
        """Get dataset for a given split, with sample transformation applied.

        !!! info

            If the a split is requested, and `subsample` is specified for that
            split, a subsample transform (via
            [`SampledDataset`][abstract_dataloader.ext.sample.]) is also
            applied.

        Args:
            split: target split.

        Returns:
            Dataset for that split, using the partially bound constructor
                passed to the `ADLDataModule`; the dataset is cached between
                calls.
        """
        if split not in self._dataset:
            raise KeyError(
                f"No `{split}` split was provided to this DataModule. Only "
                f"the following splits are present: "
                f"{list(self._dataset.keys())}")

        dataset = self._dataset[split]
        if not isinstance(dataset, spec.Dataset):
            dataset = dataset()

        subsample = self._subsample.get(split)
        if subsample is not None:
            dataset = SampledDataset(dataset, subsample)

        return TransformedDataset(dataset, transform=self.transforms.sample)

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader (`shuffle=True, drop_last=True`)."""
        return DataLoader(
            self.dataset("train"), shuffle=True, drop_last=True,
            **self._dataloader_args)

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader (`shuffle=False, drop_last=True`)."""
        return DataLoader(
            self.dataset("val"), shuffle=False, drop_last=True,
            **self._dataloader_args)

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader (`shuffle=False, drop_last=False`)."""
        return DataLoader(
            self.dataset("test"), shuffle=False, drop_last=False,
            **self._dataloader_args)
