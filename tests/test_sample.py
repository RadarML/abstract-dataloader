"""Tests for ext.sample."""

import numpy as np
import pytest
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader.ext import sample  # noqa: E402


def test_sample_ld():
    """Test low discrepancy sampler."""
    for total in [5, 20]:
        for samples in [0.5, 0.1]:
            subset = sample.sample_ld(total, samples, seed=0.0)
            assert subset.shape[0] == int(total * samples)
            assert np.all(subset >= 0)
            assert np.all(subset < total)

    subset = sample.sample_ld(20, 10, seed=42)
    assert subset.shape[0] == 10
    assert np.all(subset >= 0)
    assert np.all(subset < 20)

    with pytest.raises(ValueError):
        sample.sample_ld(20, 1.1)


def test_sampled_dataset():
    """Test for Dataset sample_ld wrapper."""

    class Dataset:
        def __getitem__(self, index):
            return index

        def __len__(self):
            return 20

    dataset = sample.SampledDataset(Dataset(), 10, seed=1, mode="ld")
    assert len(dataset) == 10
    assert dataset[0] == 1

    dataset = sample.SampledDataset(Dataset(), 0.5, mode="uniform")
    assert len(dataset) == 10
    dataset = sample.SampledDataset(Dataset(), 10, seed=0.42, mode="random")
    assert len(dataset) == 10

    # Technically illegal, but easy way to check it's being applied correctly
    dataset = sample.SampledDataset(Dataset(), 10, mode=lambda x: np.arange(3))
    assert len(dataset) == 3
