"""Locate trainer helpers in a working tree, sibling checkout, or installation.

Trainer files remain in Open4DGS-Trigs under their existing licenses; this
module only locates them. OPEN4DGS_TRIGS may name a non-sibling checkout.
"""
import importlib.util
import os
from pathlib import Path


def trigs_file(relative, core_root=None):
    root = Path(core_root) if core_root else Path(__file__).resolve().parents[1]
    explicit = os.environ.get("OPEN4DGS_TRIGS")
    candidates = ([Path(explicit) / "src/trigs"] if explicit else [])
    candidates += [root / "train/trigs/src/trigs", root.parent / "open4dgs-trigs/src/trigs"]
    for base in candidates:
        path = base / relative
        if path.is_file():
            return path
    spec = importlib.util.find_spec("trigs")
    if spec and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            path = Path(base) / relative
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"Cannot find Open4DGS-Trigs helper {relative}. Clone "
        "https://github.com/ASU-NEM-Labs/open4dgs-trigs beside open4dgs-core, "
        "set OPEN4DGS_TRIGS to its checkout, or install the trainer."
    )
