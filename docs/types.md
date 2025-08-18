# Creating a Type System

!!! abstract "Recommendations"

    - Set up a type system with [`@optree.dataclass.dataclass`es](https://optree.readthedocs.io/en/latest/dataclasses.html#optree.dataclasses.dataclass) of [jaxtyping](https://docs.kidger.site/jaxtyping/)-annotated arrays
    - Create separate protocols if cross-compatibility is required
    - Use a `optree.tree_map`-based `Collate` function

The Abstract Dataloader is built to facilitate the usage of *type systems* which describe the types (container type, numerical type, array shape, ...) that are loaded and transformed by dataloaders. While [jaxtyping](https://docs.kidger.site/jaxtyping/) is the de-facto standard for declaring array shapes and numerical types, multiple standard library alternatives exist for the container type, each with their own confusing bugs and strange limitations.

!!! question "Why Type Systems?"

    Type errors - incorrect shapes, data types, etc - are the segmentation faults of scientific computing in python. Like returning invalid pointers, returning incorrect types usually does not cause an immediate error; instead, the error only manifests when the value is used by an incompatible function such as a matrix multiplication which expects a specific shape or a function where incorrect shapes can cause memory usage to explode due to array shape broadcasting.

    Type checking helps catch and debug these errors: by checking types at logical boundaries such as function/method calls and container initialization, these errors are contained, and can be caught close to their source.

## Summary

|     | [`@dataclass`][dataclasses.dataclass] | [`TypedDict`][typing.TypedDict] | [`NamedTuple`][typing.NamedTuple] |
| --- | --- | --- | --- |
| Static type checking | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Runtime type checking | :white_check_mark: | :x: | :white_check_mark: |
| Support for generics | :white_check_mark: | :asterisk: | :asterisk: |
| Approximately a primitive type | :asterisk: | :white_check_mark: | :white_check_mark: |
| Protocol-like | :x: | :white_check_mark: | :x: |

Other, non-standard-library alternatives:

- [Pydantic](https://docs.pydantic.dev/latest/) is optimized for complex structures of simple primitives (e.g. parsing complicated JSON configurations or APIs), rather than simple structures of complex primitives (e.g. collections of numpy arrays / pytorch tensors).
- [Tensordict](https://github.com/pytorch/tensordict) appears to be [utterly broken](https://github.com/pytorch/tensordict/issues/1243) with regards to type checking. It's not clear whether this ever will (or can) be fixed.

??? info "Static type checking: it just works"

    Python type checking has come a long way, and all of these containers work out-of-the box with static type checking workflows. This includes tools like [VSCode's pylance extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (which uses [pyright](https://github.com/microsoft/pyright) under the hood), which provides type inferences when hovering over objects in the editor.

??? info "Runtime type checking: still flaky after all these years"

    Runtime type checking in python still has some way to go, and the flaws of the language and its historical baggage really show here. In particular, I'm not aware of any containers which can reliably be deeply type-checked: that is, for an arbitrary nested structure of python primitives, containers, and numpy/pytorch/etc arrays, verify the type (and shape) of each leaf value.
    
??? info "Support for generics: very helpful for numpy-pytorch cross-compatibility"

    Cross-compatibility of container types between numpy and pytorch is a huge plus, since loading data as numpy arrays and converting them to pytorch tensors is such a common design pattern. `Generic`s are the gold standard for expressing this pattern.

??? info "Approximately a primitive type: needed to work with tree libraries"

    Tree manipulation libraries such as [optree](https://github.com/metaopt/optree), `torch.utils._pytree`, `jax.tree_utils`, and the internal implementation used by `lightning.LightningModule` allow for operations to be recursively applied across "PyTrees" of various built-in container types. Notably, since a `NamedTuple` is just a fancy tuple, and a `TypedDict` is just a dict, these tree libraries all work out-of-the-box with these containers, while other containers require library-specific registration to work.

??? info "Protocol-like: in keeping with the spirit of the ADL"

    Just as the [abstract data loader's specifications][abstract_dataloader.spec] can be implemented without actually having to import the ADL, a data container can ideally be used as an interface between modules without having to agree on a common dependency. Fortunately, while container types generally don't support this, declaring a `Protocol` is an easy workaround, so this is not critical.

## `@dataclass`

!!! success "Allows deep type checking"

    The type of each attribute can be checked by runtime type checkers when the `@dataclass` is instantiated.

!!! success "Fully supports generics"

    Dataclasses can use generic types. For example:

    ```python
    @dataclass
    class Image(Generic[TArray]):
        image: UInt8[Batch, "B H W 3"]
        t: Float[TArray, "B"]
    
    # Image[np.ndarray], Image[torch.Tensor], etc.
    ```

!!! failure "Requires special handling by (some) tree methods"

    Dataclasses don't work with tree manipulation routines (which recursively crawl data structures) out of the box. In addition to libraries like [`optree`](https://github.com/metaopt/optree) or `torch.utils._pytree`, this includes [`default_collate`][torch.utils.data.default_collate] in pytorch.

    Interestingly, pytorch lightning has [silently patched this on their end](https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/apply_func.py) - probably due to the increasing popularity of dataclasses.

!!! failure "Requires a separate protocol class"
    
    Dataclasses are not "protocol-like", and two separately defined (but equivalent) dataclasses are not interchangeable. To support this, a separate protocol class needs to be defined.


To define a generic dataclass container, which can contain pytorch or numpy types:

=== "Code"

    ```python
    TArray = TypeVar("TArray", bound=torch.Tensor | np.ndarray)

    @dataclass
    class Image(Generic[TArray]):
        image: UInt8[TArray, "B H W 3"]
        t: Float[TArray, "B"]
    ```

=== "Imports"

    ```python
    # For optree compatibility, use `from optree.dataclasses import dataclass`
    from dataclasses import dataclass
    from typing import TypeVar
    import torch
    import numpy as np
    from jaxtyping import UInt8, Float
    ```

We can now declare loaders and transforms that interact with this type:

=== "Loaders"

    ```python
    class ImageSensor(spec.Sensor[Image[np.ndarray]], Metadata):
        def __getitem__(idx: int | np.integer) -> Image[np.ndarray]:
            ...
    ```

=== "Transforms"

    ```python
    takes_any_image(data: Image[TArray]) -> Image[TArray]: ...
    takes_numpy_image(data: Image[np.ndarray]) -> Image[np.ndarray]: ...
    converts_numpy_to_torch(data: Image[np.ndarray]) -> Image[torch.Tensor]: ...
    ```

We can also extend the type to add additional attributes:

```python
class DepthImage(Image[TArray]):
    depth: Float[TArray, "B H W"]

takes_any_image(DepthImage(image=..., t=..., depth=...))  # Works
```

Finally, in order to allow cross-compatibility between projects without having to share a common root class, we can instead declare a common protocol:

=== "Protocol"

    ```python
    from typing import Protocol

    @runtime_checkable
    class IsImage(Protocol, Generic[TArray]):
        image: UInt8[TArray, "B H W 3"]
        t: Float[TArray, "B"]
    ```

=== "Redefined Dataclass"

    ```python
    class ImageRedefined(Generic[TArray]):
        image: UInt8[TArray, "B H W 3"]
        t: Float[TArray, "B"]

    isinstance(ImageRedefined(image=..., t=...), IsImage)  # True
    isinstance(Image(image=..., t=...), IsImage)  # True
    ```

!!! danger "Runtime checking of protocol types may yield false positives"

    Runtime `isinstance` checks on `runtime_checkable` protocols only check that the object has all of the properties that are specified by the protocol; however, it does not verify the types of these properties. This is termed an "unsafe overlap," and the python [Protocol specification](https://typing.python.org/en/latest/spec/protocol.html#runtime-checkable-decorator-and-narrowing-types-by-isinstance) states that `isinstance` checks in type checkers should always fail in this case.
    
    Since the built-in `isinstance` does not follow this behavior, runtime type checkers (which all rely on `isinstance`) all appear to systematically ignore this.

!!! warning "Patching `default_collate` in pytorch dataloaders"

    There is currently no way to add custom node support to `torch.utils._pytree`, which is used by `default_collate`. Instead, we must provide a custom collate function with a supported pytree library, such as [optree](https://github.com/metaopt/optree).

    === "Global Registration"

        If dataclasses are registered with the optree root namespace, then the [`ext.torch.Collate`][abstract_dataloader.ext.torch.Collate] implementation which we provide is sufficient, as long as `optree` is installed.
        ```python
        from optree.dataclasses import dataclass as optree_dataclass

        dataclass = partial(optree_dataclass, namespace='')
        ```
        We also provide a pre-patched `dataclass` transform (with types set up correctly) in [`ext.types.dataclass`][abstract_dataloader.ext.types.dataclass].

    === "Namespaced Registration"

        If dataclasses are registered with a named optree namespace, then a custom collate function should be provided which uses that namespace:
        ```python
        def collate_fn(data: Sequence[TRaw]) -> TCollated:
            # Use the namespace provided to `optree.dataclasses.dataclass`
            return optree.tree_map(
                lambda *x: torch.stack(x), *data, namespace="data")
        ```

??? quote "Full example code"

    === "Without Tree Support"

        ```python
        from dataclasses import dataclass
        from typing import Protocol, TypeVar

        import torch
        import numpy as np
        from jaxtyping import UInt8, Float

        TArray = TypeVar("TArray", bound=torch.Tensor | np.ndarray)

        @dataclass
        class Image(Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]

        @dataclass
        class DepthImage(Image[TArray]):
            depth: Float[TArray, "B H W"]

        @runtime_checkable
        class IsImage(Protocol, Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]
        ```

    === "Optree (Root Namespace)"

        ```python
        from functools import partial
        from typing import Protocol, TypeVar

        from optree.dataclasses import dataclass as optree_dataclass
        from optree.registry import __GLOBAL_NAMESPACE
        import torch
        import numpy as np
        from jaxtyping import UInt8, Float

        dataclass = partial(optree_dataclass, namespace=__GLOBAL_NAMESPACE)

        TArray = TypeVar("TArray", bound=torch.Tensor | np.ndarray)

        @dataclass
        class Image(Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]

        @dataclass
        class DepthImage(Image[TArray]):
            depth: Float[TArray, "B H W"]

        @runtime_checkable
        class IsImage(Protocol, Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]
        ```

    === "Optree (User Namespace)"

        ```python
        from optree.dataclasses import dataclass
        from typing import Protocol, TypeVar

        import torch
        import numpy as np
        from jaxtyping import UInt8, Float

        TArray = TypeVar("TArray", bound=torch.Tensor | np.ndarray)

        @dataclass(namespace="data")
        class Image(Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]

        @dataclass(namespace="data")
        class DepthImage(Image[TArray]):
            depth: Float[TArray, "B H W"]

        @runtime_checkable
        class IsImage(Protocol, Generic[TArray]):
            image: UInt8[TArray, "B H W 3"]
            t: Float[TArray, "B"]
        ```

## `TypedDict`

!!! success "Works with tree libraries out of the box"

    Since [`TypedDict`][typing.TypedDict] are just dictionaries, they work with tree manipulation routines out of the box.

!!! success "Natively Protocol-like"

    `TypedDict` are just annotations, and behave like protocols: separately defined `TypedDict` with identical specifications can be used interchangeably. This removes the need to define a separate container type and protocol type.

!!! failure ":rotating_light: Fundamentally broken and not runtime type checkable :rotating_light:"

    While the `TypedDict` spec provides type checking on paper, `isinstance` checks are forbidden, which in practice [makes runtime type checking of `TypedDict`s impossible](https://github.com/beartype/beartype/issues/55), since all runtime type-checkers rely on `isinstance`. This is a problem, since the entire point of typed data containers is to facilitate runtime type checking of array shapes!

    Generic `TypedDict`s are also [supported in the spec](https://github.com/python/cpython/issues/89026); however, due to forbidding `isinstance` checks, they [cause even more problems for runtime type checkers](https://github.com/beartype/beartype/issues/519#issuecomment-2831892484).

    In practice, this means runtime type checkers like beartype fall back to using `Mapping[str, Any]` or even just `dict` when they encounter a `TypedDict` (and completely explode when they encounter a generic `TypedDict`). Unfortunately, since the problems with `TypedDict` originate from fundamental design choices in python's type system, it's unclear if this will ever be fixed -- or if this even can be fixed.


Defining a `TypedDict`-based container which supports both numpy arrays and pytorch tensors requires defining separate classes:

```python
from typing import TypedDict

import torch
import numpy as np
from jaxtyping import UInt8, Float

class ImageTorch(TypedDict):
    image: UInt8[torch.Tensor, "B H W 3"]
    t: Float[torch.Tensor, "B"]

class ImageNP(TypedDict):
    image: UInt8[np.ndarray, "B H W 3"]
    t: Float[np.ndarray, "B"]
```

... and that's it. Everything that can work will work out of the box, but no amount of workarounds will ever make what doesn't work, work.

## `NamedTuple`

!!! success "Allows deep type checking"

    The type of each attribute can be checked by runtime type checkers when the `NamedTuple` is instantiated.

!!! success "Works with tree libraries out of the box"

    Since [`TypedDict`][typing.TypedDict] are just dictionaries, they work with tree manipulation routines out of the box.

!!! failure "Buggy Generics"

    While not as much of a disaster as `TypedDict`, the inheritance rules for `NamedTuple` also make it tricky for runtime type checkers to [properly support generics](https://github.com/beartype/beartype/issues/519).

!!! failure "Requires a separate protocol class"
    
    Dataclasses are not "protocol-like", and two separately defined (but equivalent) dataclasses are not interchangeable. To support this, a separate protocol class needs to be defined.

Like `TypedDict`, we need to define separate containers which supports both numpy arrays and pytorch tensors:

```python
from typing import NamedTuple

import torch
import numpy as np
from jaxtyping import UInt8, Float

class ImageTorch(NamedTuple):
    image: UInt8[torch.Tensor, "B H W 3"]
    t: Float[torch.Tensor, "B"]

class ImageNP(NamedTuple):
    image: UInt8[np.ndarray, "B H W 3"]
    t: Float[np.ndarray, "B"]
```

Annoyingly, we now have to also define a matching set of `Protocol`s for both versions:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class IsImageTorch(Protocol):
    image: UInt8[torch.Tensor, "B H W 3"]
    t: Float[torch.Tensor, "B"]

@runtime_checkable
class IsImageNP(Protocol):
    image: UInt8[np.ndarray, "B H W 3"]
    t: Float[np.ndarray, "B"]
```
