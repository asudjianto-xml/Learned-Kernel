"""lkbook — companion code for *The Learned Kernel* (Agus Sudjianto).

Running-example data loaders and plotting style shared across chapters, plus one
module per chapter under ``lkbook.chapters``. Imported by the chapter notebooks so
the notebook numbers and figures match the printed book exactly.
"""
from .data import RunningData, TabularData, load_california, load_taiwan, load_bikeshare
from .plotting import set_style, SIGNED_CMAP, POS_CMAP

__version__ = "0.12.0"
__all__ = [
    "RunningData", "TabularData", "load_california", "load_taiwan", "load_bikeshare",
    "set_style", "SIGNED_CMAP", "POS_CMAP", "__version__",
]
