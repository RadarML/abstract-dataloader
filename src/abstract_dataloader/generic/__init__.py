"""Generic "ready-to-use" implementations of common components.

Other generic and largely reusable components can be added to this submodule.

!!! note

    Numpy (and jaxtyping) are the only dependencies; to keep the
    `abstract_dataloader`'s dependencies lightweight and flexible, components
    should only be added here if they do not require any additional
    dependencies.
"""

from .composition import DatasetMeta, ParallelPipelines, ParallelTransforms
from .sequence import Metadata, SequencePipeline, Window
from .sync import Decimate, Empty, Nearest, Next

__all__ = [
    "DatasetMeta", "ParallelPipelines", "ParallelTransforms",
    "Metadata", "SequencePipeline", "Window",
    "Decimate", "Empty", "Nearest", "Next"
]
