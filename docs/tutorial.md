# Using an Abstract Dataloader

A fully implemented Abstract Dataloader (ADL) compliant system should consist of a collection of modular components implementing sensor data reading, time synchronization, trace & dataset handling, followed by data preprocessing.

![Abstract Dataloader Overview](./diagrams/overview.svg)

In this tutorial, we cover how to use these components if you are given implementations for some or all of them; we split this into data loading and processing, which are connected only by a trivial interface &mdash; the output of the [`Dataset`](#dataset) should be the input of the [`Pipeline`](#pipeline).


## Dataset

At a minimum, any ADL-compliant dataloader should include one or more [`Sensor`][abstract_dataloader.spec.Sensor] implementations. These may be accompanied by custom [`Synchronization`][abstract_dataloader.spec.Synchronization], [`Trace`][abstract_dataloader.spec.Trace], and [`Dataset`][abstract_dataloader.spec.Dataset] implementations.

### [`Sensor`][abstract_dataloader.spec.Sensor]

Sensors must implementing the [`Sensor`][abstract_dataloader.spec.Sensor] specification:

- Sensors have a [`metadata: spec.Metadata`][abstract_dataloader.spec.Metadata] attribute, which contains a `timestamps: Float64[np.ndarray, "N"]` attribute.
- Each sensor has a `__getitem__` which can be used to read data by index, and a `__len__`.

!!! warning

    The ADL does not prescribe a standardized API for initializing a `Sensor` since this is highly implementation-specific.

### [`Trace`][abstract_dataloader.spec.Trace]

Once you have initialized a collection of sensors which all correspond to simultaneously-recorded sensor data, assemble them into a [`Trace`][abstract_dataloader.spec.Trace]:

- `Trace` implementations supplied by an ADL-compliant dataloader may have their own initialization methods.
- A trace has a `__getitem__` which reads data from each sensor corresponding to some global index assigned by a [`Synchronization`][abstract_dataloader.spec.Synchronization] policy, and a `__len__`.

In the case that a data loading library does not provide a custom `Trace` implementation, [`abstract_dataloader.abstract.Trace`][abstract_dataloader.abstract.Trace] can be used instead as a simple, no-frills baseline implementation.

!!! info

    A [`Trace`][abstract_dataloader.spec.Trace] implementation should take (and [`abstract.Trace`][abstract_dataloader.abstract.Trace] does take) a [`Synchronization`][abstract_dataloader.spec.Synchronization] policy as an argument. While a particular ADL-compliant dataloader implementation may provide custom `Synchronization` policies, a few generic implementations are included with [`abstract_dataloader.generic`][abstract_dataloader.generic]:

    | Class | Description |
    | ----- | ----------- |
    | [`Empty`][abstract_dataloader.generic.Empty] | a no-op for intializing a trace without any synchronization (i.e., just as a container of sensors). |
    | [`Nearest`][abstract_dataloader.generic.Nearest] | find the nearest measurement for each sensor relative to the reference sensor's measurements. |
    | [`Next`][abstract_dataloader.generic.Next] | find the next measurement for each sensor relative to the reference sensor's measurements. |

### [`Dataset`][abstract_dataloader.spec.Dataset]

Finally, once a collection of [`Trace`][abstract_dataloader.spec.Trace] objects are initialized, combine them into a [`Dataset`][abstract_dataloader.spec.Dataset].

- As with `Trace`, `Dataset` implementations supplied by an ADL-compliant dataloader may have their own initialization methods.
- Similarly, datasets have a  `__getitem__` which reads data from each sensor and a `__len__`.

In the case that a data loading library does not provide a custom `Dataset` implementation, [`abstract_dataloader.abstract.Dataset`][abstract_dataloader.abstract.Dataset] can also be used instead.

!!! tip

    If your use case does not require combining multiple traces into a dataset, you can directly use a `Trace` as a `Dataset`: the `Trace` protocol *subclasses* the `Dataset` protocol, in that all required interfaces are a subset of the `Dataset` interfaces.

## Pipeline

Dataloaders may or may not come with data a preprocessing [`Pipeline`][abstract_dataloader.spec.Pipeline] which you are expected to use. Since data loaders ([`Dataset`][abstract_dataloader.spec.Dataset]) and processing pipelines ([`Pipeline`][abstract_dataloader.spec.Pipeline]) are modular and freely composable assuming they share the same data types, it's also possible that a pipeline is distributed separately from the data loader(s) which it is compatible with.

### Use Out-of-the Box

If a library comes with a complete, ready-to-use `Pipeline`, then all that remains is to apply the pipeline:

```python
def apply_pipeline(indices: Sequence[int], dataset: Dataset, pipeline: Pipeline):
    raw = [dataset[i] for i in indices]
    transformed = [pipeline.sample(x) for x in raw]
    collated = pipeline.collate(transformed)
    processed = pipeline.batch(collated)
    return processed
```

!!! tip

    In practice, the pipeline should be integrated with the data loading and training pipeline to ensure that it is properly pipelined and parallelized!

    Assuming you are using pytorch, the following building blocks may be helpful:

    - Use [`TransformedDataset`][abstract_dataloader.ext.torch.TransformedDataset], which provides a pytorch map-style dataset with a pipeline's `.transform` applied.
    - Pass `Pipeline.collate` as the `collate_fn` for a pytorch [`DataLoader`][torch.utils.data.DataLoader].
    - If you are using pytorch lightning, [`ext.lightning.ADLDataModule`][abstract_dataloader.ext.lightning.ADLDataModule] can also handle `.transform`, `.collate`, and a number of other common data marshalling steps for you.

### Assemble from Components

If you don't have a complete `Pipeline` implementation but instead have separate components, you can use [`abstract.Pipeline`][abstract_dataloader.abstract.Pipeline] to assemble a `transform`: [`spec.Transform`][abstract_dataloader.spec.Transform], `collate`: [`spec.Collate`][abstract_dataloader.spec.Collate], and `batch`: [`spec.Transform`][abstract_dataloader.spec.Transform] into a single pipeline.

!!! tip

    Both `transform` and `pipeline` are optional and will fall back to the identity function if not provided. Only `collate` is required.

!!! info

    For pytorch users, a reference collate function is provided as [`abstract_dataloader.ext.torch.Collate`][abstract_dataloader.ext.torch.Collate], which can handle common pytorch tensor operations and data structures.

    If your `Pipeline` contains any [`torch.nn.Module`][torch.nn.Module]s, you may find [`abstract_dataloader.ext.torch.Pipeline`][abstract_dataloader.ext.torch.Pipeline] helpful, which will automatically register these for you.
