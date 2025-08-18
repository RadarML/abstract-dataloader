::: abstract_dataloader.ext

Unlike [`abstract_dataloader.generic`][abstract_dataloader.generic], these implementations "extend" the ADL spec by imposing a particular conceptual framework on various functionality.

!!! warning

    This module and its submodules are not automatically imported; you will need to explicitly import them:
    ```python
    from abstract_dataloader.ext import torch
    ```

!!! info

    Not all extension modules are included in the test suite or CI, and these moduels are generally held to a lower standard than the core `abstract_dataloader`.

## Extension Modules

<div class="grid cards" markdown>

- [`augment`][abstract_dataloader.ext.augment]

    A protocol for specifying data augmentations.

- [`graph`][abstract_dataloader.ext.graph]

    A programming model for composing a DAG of callables into a single transform.

- [`lightning`][abstract_dataloader.ext.lightning]

    A lightning datamodule wrapper for ADL datasets and pipelines.

- [`objective`][abstract_dataloader.ext.objective]

    A programming model for training objectives and multi-objective learning.

- [`sample`][abstract_dataloader.ext.sample]

    Dataset sampling utilities, including a low-discrepancy subset sampler.

- [`torch`][abstract_dataloader.ext.torch]

    Pytorch-specific utilities, including a collate function and a `torch.nn.Module` pipeline.

- [`types`][abstract_dataloader.ext.types]

    Type-related utilities which are not part of the core ADL spec.

</div>
