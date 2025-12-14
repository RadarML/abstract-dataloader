"""Tests for ext.augment."""

import numpy as np
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader.ext import augment  # noqa: E402


def test_bernoulli():
    """Test Bernoulli augmentation."""
    rng = np.random.default_rng(42)
    assert augment.Bernoulli(p=1.0)(rng) is True
    assert augment.Bernoulli(p=0.0)(rng) is False


def test_normal():
    """Test Normal augmentation."""
    rng = np.random.default_rng(42)

    # Basic functionality
    result = augment.Normal(p=1.0, std=2.0)(rng)
    assert isinstance(result, float)

    # Probability gating
    assert augment.Normal(p=0.0, std=2.0)(rng) == 0.0


def test_truncated_log_normal():
    """Test TruncatedLogNormal augmentation."""
    rng = np.random.default_rng(42)

    # Basic functionality
    result = augment.TruncatedLogNormal(p=1.0)(rng)
    assert isinstance(result, float) and result > 0

    # Probability gating
    assert augment.TruncatedLogNormal(p=0.0)(rng) == 1.0


def test_uniform():
    """Test Uniform augmentation."""
    rng = np.random.default_rng(42)

    # Basic functionality
    result = augment.Uniform(p=1.0, lower=-5.0, upper=5.0)(rng)
    assert isinstance(result, float) and -5.0 <= result <= 5.0

    # Probability gating
    assert augment.Uniform(p=0.0)(rng) == 0.0


def test_augmentations_collection():
    """Test Augmentations collection class."""
    augs = augment.Augmentations(
        flip=augment.Bernoulli(p=1.0),
        scale=augment.Normal(p=1.0, std=0.1)
    )

    # Basic properties
    assert len(augs.augmentations) == 2
    assert isinstance(augs.rng, np.random.Generator)
    assert len(list(augs.children())) == 2

    # Training mode returns values
    result = augs(meta={"train": True})
    assert len(result) == 2
    assert isinstance(result["flip"], bool)
    assert isinstance(result["scale"], float)

    # Inference mode returns empty
    assert augs(meta={"train": False}) == {}

    # Empty augmentations
    empty_augs = augment.Augmentations()
    assert empty_augs() == {}
    assert len(list(empty_augs.children())) == 0
