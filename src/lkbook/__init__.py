"""lkbook — companion code for *The Learned Kernel* (Agus Sudjianto).

Running-example data loaders and plotting style shared across chapters, plus one
module per chapter under ``lkbook.chapters``. Imported by the chapter notebooks so
the notebook numbers and figures match the printed book exactly.
"""
from .data import RunningData, load_california, load_taiwan
from .plotting import set_style, SIGNED_CMAP, POS_CMAP

__version__ = "0.1.0"
__all__ = [
    "RunningData", "load_california", "load_taiwan",
    "set_style", "SIGNED_CMAP", "POS_CMAP", "__version__",
]
