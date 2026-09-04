"""Persistent library indices for Flash-based spectral similarity search."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


_FLASH_INDEX_FORMAT = "matchms.flash_index"
_FLASH_INDEX_VERSION = 1


@dataclass
class FlashIndex:
    """Reusable library index for Flash-based spectral similarity methods.

    The arrays stored here are the library-side data required by the Flash
    workers.  Keeping them in a public container makes the expensive library
    preprocessing/index construction independent from individual ``matrix``
    calls and allows the index to be saved and loaded.

    Notes
    -----
    ``config`` contains only parameters that affect construction of the index.
    Search-time parameters such as fragment tolerance and identity precursor
    tolerance are intentionally not part of the index configuration.
    """

    n_specs: int
    dtype: np.dtype
    peaks_mz: np.ndarray
    peaks_int: np.ndarray
    peaks_spec_idx: np.ndarray
    spec_offsets: np.ndarray
    spec_mz: np.ndarray
    spec_int: np.ndarray
    precursor_mz: np.ndarray
    spec_l2: np.ndarray | None = None
    nl_mz: np.ndarray | None = None
    nl_spec_idx: np.ndarray | None = None
    nl_product_idx: np.ndarray | None = None
    config: dict[str, Any] | None = None

    @classmethod
    def from_library(cls, library, *, config: dict[str, Any]) -> "FlashIndex":
        """Create a public index from the internal Flash library object."""
        return cls(
            n_specs=int(library.n_specs),
            dtype=np.dtype(library.dtype),
            peaks_mz=library.peaks_mz,
            peaks_int=library.peaks_int,
            peaks_spec_idx=library.peaks_spec_idx,
            spec_offsets=library.spec_offsets,
            spec_mz=library.spec_mz,
            spec_int=library.spec_int,
            precursor_mz=library.precursor_mz,
            spec_l2=getattr(library, "spec_l2", None),
            nl_mz=getattr(library, "nl_mz", None),
            nl_spec_idx=getattr(library, "nl_spec_idx", None),
            nl_product_idx=getattr(library, "nl_product_idx", None),
            config=dict(config),
        )

    def save(self, filename: str | Path) -> None:
        """Save this index to a versioned ``.npz`` file.

        The file contains only NumPy arrays plus JSON metadata and therefore
        does not rely on pickle.  Saving is atomic on filesystems that support
        ``os.replace``.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "format": _FLASH_INDEX_FORMAT,
            "version": _FLASH_INDEX_VERSION,
            "n_specs": int(self.n_specs),
            "dtype": self.dtype.str,
            "config": self.config or {},
            "optional_arrays": {
                "spec_l2": self.spec_l2 is not None,
                "nl_mz": self.nl_mz is not None,
                "nl_spec_idx": self.nl_spec_idx is not None,
                "nl_product_idx": self.nl_product_idx is not None,
            },
        }

        arrays = {
            "__metadata__": np.asarray(json.dumps(metadata)),
            "peaks_mz": self.peaks_mz,
            "peaks_int": self.peaks_int,
            "peaks_spec_idx": self.peaks_spec_idx,
            "spec_offsets": self.spec_offsets,
            "spec_mz": self.spec_mz,
            "spec_int": self.spec_int,
            "precursor_mz": self.precursor_mz,
        }
        if self.spec_l2 is not None:
            arrays["spec_l2"] = self.spec_l2
        if self.nl_mz is not None:
            arrays["nl_mz"] = self.nl_mz
        if self.nl_spec_idx is not None:
            arrays["nl_spec_idx"] = self.nl_spec_idx
        if self.nl_product_idx is not None:
            arrays["nl_product_idx"] = self.nl_product_idx

        tmp = path.with_name(path.name + ".tmp")
        try:
            # Using a file handle prevents np.savez from silently appending
            # '.npz' to paths with another suffix.
            with tmp.open("wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, filename: str | Path) -> "FlashIndex":
        """Load a :class:`FlashIndex` previously written by :meth:`save`."""
        path = Path(filename)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["__metadata__"].item()))

            if metadata.get("format") != _FLASH_INDEX_FORMAT:
                raise ValueError(
                    f"Not a matchms Flash index: {metadata.get('format')!r}."
                )
            if metadata.get("version") != _FLASH_INDEX_VERSION:
                raise ValueError(
                    "Unsupported Flash index version "
                    f"{metadata.get('version')!r}; expected {_FLASH_INDEX_VERSION}."
                )

            optional = metadata.get("optional_arrays", {})

            def optional_array(name: str):
                if not optional.get(name, False):
                    return None
                return np.asarray(archive[name])

            return cls(
                n_specs=int(metadata["n_specs"]),
                dtype=np.dtype(metadata["dtype"]),
                peaks_mz=np.asarray(archive["peaks_mz"]),
                peaks_int=np.asarray(archive["peaks_int"]),
                peaks_spec_idx=np.asarray(archive["peaks_spec_idx"]),
                spec_offsets=np.asarray(archive["spec_offsets"]),
                spec_mz=np.asarray(archive["spec_mz"]),
                spec_int=np.asarray(archive["spec_int"]),
                precursor_mz=np.asarray(archive["precursor_mz"]),
                spec_l2=optional_array("spec_l2"),
                nl_mz=optional_array("nl_mz"),
                nl_spec_idx=optional_array("nl_spec_idx"),
                nl_product_idx=optional_array("nl_product_idx"),
                config=dict(metadata.get("config", {})),
            )

    @property
    def has_neutral_loss_index(self) -> bool:
        """Whether neutral-loss search arrays are present."""
        return self.nl_mz is not None

    def __repr__(self) -> str:
        return (
            f"FlashIndex(n_specs={self.n_specs}, dtype={self.dtype}, "
            f"neutral_loss={self.has_neutral_loss_index})"
        )
