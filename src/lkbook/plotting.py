"""Shared plotting style for *The Learned Kernel* figures and notebooks."""
from __future__ import annotations

import matplotlib as mpl

# diverging map for signed influence, sequential for nonnegative influence
SIGNED_CMAP = "RdBu_r"
POS_CMAP = "Reds"


def set_style() -> None:
    """House matplotlib defaults: clean, readable, book-consistent."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "image.aspect": "auto",
    })
