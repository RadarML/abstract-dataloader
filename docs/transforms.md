# Data Transforms

!!! abstract "TL;DR"

    - A [`Transform`][abstract_dataloader.spec.Transform] is any data transformation.
    - [`Collate`][abstract_dataloader.spec.Collate] is a special transform which aggregates a [`Sequence`][typing.Sequence] of data into a batch.
    - Combining a CPU-side [`Transform`][abstract_dataloader.spec.Transform], a [`Collate`][abstract_dataloader.spec.Collate] function, and a GPU-side [`Transform`][abstract_dataloader.spec.Transform] yields a data [`Pipeline`][abstract_dataloader.spec.Pipeline].

## Transforms

Since data processing steps can vary wildly between domains, the lowest common denominator to describe a data transformation is simply a `Callable[[TRaw], TTransformed]`: a callable which takes an input data type, and converts it to some other data type. We provide this as the suggestively-named `Transform` protocol:

```python
class Transform(Protocol, Generic[TRaw, TTransformed]):
    """Sample or batch-wise transform."""

    def __call__(self, data: TRaw) -> TTransformed:
        """Apply transform to a single sample."""
        ...
```

## Batching and Collation

While clearly universal, the "all data processing is composed callables" is too vague, and not really helpful for organizing data transforms. To better categorize data transforms, we turn to analyzing batched operation.

From a code portability standpoint, we can break down all transforms based on whether they support batched operation in the inputs and/or outputs. This implies that there are four possible types of transforms: single-sample to single-sample, single-samples to batch, batch to batch, and batch to single-sample

!!! question

    Are there any use cases for batch to single-sample transforms? I'm not aware of any, though perhaps there are some edge cases out there.

Within the first three (which are commonly used) the single-sample to batch transform stands out. We define this as a narrower type compared to a generic transform, which we refer to as [`Collate`][abstract_dataloader.spec.Collate], which is analogous to the `collate_fn` of a [pytorch dataloader](https://pytorch.org/docs/stable/data.html):

```
class Collate(Protocol, Generic[TTransformed, TCollated]):
    """Data collation."""

    def __call__(self, data: Sequence[TTransformed]) -> TCollated:
        """Collate a set of samples."""
        ...
```

## Pipelines

A typical data processing pipeline consists of a CPU-side transform, a batching function, and a GPU-side transform. We formalize this using a [`Pipeline`][abstract_dataloader.spec.Pipeline], which collects these three components together into a generically typed container:

![Data preprocessing programming model](diagrams/pipeline.svg)

- [`Pipeline.sample`][abstract_dataloader.spec.Pipeline.sample]: apply some transform to a single sample, returning another single sample. This represents most common dataloader operations, e.g. data augmentations, point cloud processing, and nominally occurs on the CPU.
- [`Pipeline.collate`][abstract_dataloader.spec.Pipeline.collate]: combine multiple samples into a "batch" which facilitates vectorized processing.
- [`Pipeline.batch`][abstract_dataloader.spec.Pipeline.batch]: this step operates solely on batched data; there is no sharp line where GPU preprocessing ends, and a GPU-based model begins. This captures expensive, GPU-accelerated preprocessing steps.

!!! note

    Since the only distinction between sample-to-sample and batch-to-batch transforms is the definition of a sample and a batch, which are inherently domain and implementation specific, we use the same generic [`Transform`][abstract_dataloader.spec.Transform] type for both [`Pipeline.sample`][abstract_dataloader.spec.Pipeline.sample] and [`Pipeline.batch`][abstract_dataloader.spec.Pipeline.batch], unlike [`Pipeline.collate`][abstract_dataloader.spec.Pipeline.collate], which accepts the distinct [`Collate`][abstract_dataloader.spec.Collate] type.
    
    Implementations may also be `.sample`/`.batch`-generic, for example using overloaded operators only, operating on pytorch `cpu` and `cuda` tensors, or having a `numpy`/`pytorch` switch. As such, we leave distinguishing `.sample` and `.batch` transforms up to the user.
