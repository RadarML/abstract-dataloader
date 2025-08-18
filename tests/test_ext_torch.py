"""Tests for ext.torch."""

from typing import Generic

import numpy as np
import torch
from beartype.claw import beartype_package

beartype_package("abstract_dataloader")

from abstract_dataloader.ext import graph, types  # noqa: E402
from abstract_dataloader.ext.torch import (  # noqa: E402
    Collate,
    Pipeline,
    TransformedDataset,
)


def test_collate_stack():
    """Test collate function."""
    collate = Collate(mode='stack')

    # Test with a simple list of tensors
    data = [np.array([1, 2]), np.array([3, 4])]
    result = collate(data)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[1, 2], [3, 4]]))


def test_collate_concat():
    """Test collate function with concatenation."""
    collate = Collate(mode='concat')

    # Test with a simple list of tensors
    data = [np.array([1, 2])[None], np.array([3, 4])[None]]
    result = collate(data)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[1, 2], [3, 4]]))


def test_collate_nested_dict_of_dataclasses():
    """Test collate function with a nested dict of dataclasses of arrays."""
    @types.dataclass
    class DataClassExample(Generic[types.TArray]):
        array: types.TArray

    collate = Collate(mode='stack')

    data = [
        {'key': DataClassExample(array=np.array([1, 2]))},
        {'key': DataClassExample(array=np.array([3, 4]))}
    ]

    result = collate(data)
    assert isinstance(result, dict)
    assert isinstance(result['key'], DataClassExample)
    assert torch.allclose(result['key'].array, torch.tensor([[1, 2], [3, 4]]))


def test_transformed_dataset():
    """Test transformed dataset."""

    class Dataset:
        def __getitem__(self, index):
            return index

        def __len__(self):
            return 3

    # Dummy transform
    def tf(data):
        return data * 2

    dataset = Dataset()
    transformed_dataset = TransformedDataset(dataset, transform=tf)

    # Test length
    assert len(transformed_dataset) == len(dataset)

    # Test transformed data
    for i in range(len(transformed_dataset)):
        assert transformed_dataset[i] == tf(dataset[i])


def test_torch_pipeline():
    """Test module discovery for torch-compatible Pipeline."""
    def sample_fn(data):
        return data  # pragma: no cover

    class Module(torch.nn.Module):
        def forward(self, batch):
            return batch  # pragma: no cover

    # Simple case: module directly as batch
    module = Module()
    pipeline = Pipeline(sample=sample_fn, collate=Collate(), batch=module)
    assert module in list(pipeline.submodules.children())

    # More complicated case: batch is a Transform containing the Module
    transform_batch = graph.Transform(
        outputs={'result': 'processed'},
        process={
            'transform': module,
            'output': 'processed',
            'inputs': {'batch': 'input'}
        }
    )
    pipeline_with_transform = Pipeline(
        sample=sample_fn, collate=Collate(), batch=transform_batch)

    # Check that the module within the Transform is discovered
    children_list = list(pipeline_with_transform.submodules.children())
    assert len(children_list) == 1
    assert module in children_list
