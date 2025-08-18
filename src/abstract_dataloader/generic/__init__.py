"""Generic "ready-to-use" implementations of common components.

Other generic and largely reusable components can be added to this submodule.

!!! note

    Numpy (and jaxtyping) are the only dependencies; to keep the
    `abstract_dataloader`'s dependencies lightweight and flexible, components
    should only be added here if they do not require any additional
    dependencies.
"""

from .composition import ParallelPipelines, ParallelTransforms
from .sequence import Metadata, SequencePipeline, Window
from .sync import Empty, Nearest, Next

__all__ = [
    "ParallelPipelines", "ParallelTransforms",
    "Metadata", "SequencePipeline", "Window",
    "Empty", "Nearest", "Next"
]
