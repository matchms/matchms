"""Persistent indices for SpectraCollection-native Flash similarity search."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_FLASH_INDEX_FORMAT = "matchms.flash_index"
_FLASH_INDEX_VERSION = 2


@dataclass
class FlashIndex:
    """Reusable library index for SpectraCollection-native Flash search.

    This is the persistent/public counterpart of the internal ``_LibraryIndex``
    produced by :func:`matchms.similarity.flash_utils._build_library_index_from_prepared`.
    It intentionally exposes the same array attributes so the existing Flash row
    workers can use a ``FlashIndex`` directly without conversion.

    ``config`` stores only parameters that affect preprocessing/index construction.
    Search-time parameters such as fragment tolerance, precursor tolerance,
    score threshold, and top-k are deliberately excluded so one saved index can be
    reused for many searches.
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
    nl_int: np.ndarray | None = None
    nl_spec_idx: np.ndarray | None = None
    nl_product_idx: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_specs = int(self.n_specs)
        self.dtype = np.dtype(self.dtype)
        self.config = dict(self.config or {})
        self.metadata = dict(self.metadata or {})
        self._validate_arrays()

    @classmethod
    def from_library(
        cls,
        library,
        *,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "FlashIndex":
        """Wrap an internal SpectraCollection-native ``_LibraryIndex``.

        No peak arrays are copied. The expensive preprocessing and global sorting
        have already happened in ``_prepare_collection`` and
        ``_build_library_index_from_prepared``.
        """
        return cls(
            n_specs=library.n_specs,
            dtype=library.dtype,
            peaks_mz=library.peaks_mz,
            peaks_int=library.peaks_int,
            peaks_spec_idx=library.peaks_spec_idx,
            spec_offsets=library.spec_offsets,
            spec_mz=library.spec_mz,
            spec_int=library.spec_int,
            precursor_mz=library.precursor_mz,
            spec_l2=getattr(library, "spec_l2", None),
            nl_mz=getattr(library, "nl_mz", None),
            nl_int=getattr(library, "nl_int", None),
            nl_spec_idx=getattr(library, "nl_spec_idx", None),
            nl_product_idx=getattr(library, "nl_product_idx", None),
            config=config,
            metadata=metadata or {},
        )

    @property
    def has_neutral_loss_index(self) -> bool:
        """Return whether neutral-loss search arrays are available."""
        return self.nl_mz is not None

    @property
    def has_l2_norms(self) -> bool:
        """Return whether per-spectrum L2 norms are available."""
        return self.spec_l2 is not None

    def save(self, filename: str | Path) -> None:
        """Save the index as a versioned, non-pickle ``.npz`` file.

        ``np.savez`` is intentionally used instead of compressed NPZ. For very
        large Flash indices this prioritizes write/load speed over disk space.
        The write uses a temporary file followed by ``os.replace``.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "format": _FLASH_INDEX_FORMAT,
            "version": _FLASH_INDEX_VERSION,
            "n_specs": self.n_specs,
            "dtype": self.dtype.str,
            "config": self.config,
            "metadata": self.metadata,
            "optional_arrays": {
                "spec_l2": self.spec_l2 is not None,
                "nl_mz": self.nl_mz is not None,
                "nl_int": self.nl_int is not None,
                "nl_spec_idx": self.nl_spec_idx is not None,
                "nl_product_idx": self.nl_product_idx is not None,
            },
        }

        arrays = {
            "__metadata__": np.asarray(json.dumps(meta)),
            "peaks_mz": self.peaks_mz,
            "peaks_int": self.peaks_int,
            "peaks_spec_idx": self.peaks_spec_idx,
            "spec_offsets": self.spec_offsets,
            "spec_mz": self.spec_mz,
            "spec_int": self.spec_int,
            "precursor_mz": self.precursor_mz,
        }
        for name in (
            "spec_l2",
            "nl_mz",
            "nl_int",
            "nl_spec_idx",
            "nl_product_idx",
        ):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value

        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, filename: str | Path) -> "FlashIndex":
        """Load a :class:`FlashIndex` written by :meth:`save`."""
        path = Path(filename)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["__metadata__"].item()))

            if meta.get("format") != _FLASH_INDEX_FORMAT:
                raise ValueError(
                    f"Not a matchms Flash index: {meta.get('format')!r}."
                )
            version = meta.get("version")
            if version != _FLASH_INDEX_VERSION:
                raise ValueError(
                    f"Unsupported Flash index version {version!r}; "
                    f"expected {_FLASH_INDEX_VERSION}."
                )

            optional = meta.get("optional_arrays", {})

            def get_optional(name: str):
                if not optional.get(name, False):
                    return None
                if name not in archive:
                    raise ValueError(
                        f"Flash index metadata declares array {name!r}, "
                        "but the array is missing from the file."
                    )
                return np.asarray(archive[name])

            return cls(
                n_specs=meta["n_specs"],
                dtype=np.dtype(meta["dtype"]),
                peaks_mz=np.asarray(archive["peaks_mz"]),
                peaks_int=np.asarray(archive["peaks_int"]),
                peaks_spec_idx=np.asarray(archive["peaks_spec_idx"]),
                spec_offsets=np.asarray(archive["spec_offsets"]),
                spec_mz=np.asarray(archive["spec_mz"]),
                spec_int=np.asarray(archive["spec_int"]),
                precursor_mz=np.asarray(archive["precursor_mz"]),
                spec_l2=get_optional("spec_l2"),
                nl_mz=get_optional("nl_mz"),
                nl_int=get_optional("nl_int"),
                nl_spec_idx=get_optional("nl_spec_idx"),
                nl_product_idx=get_optional("nl_product_idx"),
                config=meta.get("config", {}),
                metadata=meta.get("metadata", {}),
            )

    def _validate_arrays(self) -> None:
        """Validate structural invariants expected by the Flash workers."""
        required = {
            "peaks_mz": self.peaks_mz,
            "peaks_int": self.peaks_int,
            "peaks_spec_idx": self.peaks_spec_idx,
            "spec_offsets": self.spec_offsets,
            "spec_mz": self.spec_mz,
            "spec_int": self.spec_int,
            "precursor_mz": self.precursor_mz,
        }
        for name, value in required.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"FlashIndex.{name} must be a NumPy array.")

        if self.n_specs < 0:
            raise ValueError("FlashIndex.n_specs must be >= 0.")
        if self.spec_offsets.ndim != 1 or self.spec_offsets.size != self.n_specs + 1:
            raise ValueError("spec_offsets must have length n_specs + 1.")
        if self.spec_offsets.size and int(self.spec_offsets[0]) != 0:
            raise ValueError("spec_offsets must start at 0.")
        if np.any(np.diff(self.spec_offsets) < 0):
            raise ValueError("spec_offsets must be monotonically non-decreasing.")
        if self.spec_mz.ndim != 1 or self.spec_int.ndim != 1:
            raise ValueError("spec_mz and spec_int must be 1D arrays.")
        if self.spec_mz.size != self.spec_int.size:
            raise ValueError("spec_mz and spec_int must have identical lengths.")
        if self.spec_offsets.size and int(self.spec_offsets[-1]) != self.spec_mz.size:
            raise ValueError("spec_offsets[-1] must equal the number of spectrum-major peaks.")
        if self.precursor_mz.ndim != 1 or self.precursor_mz.size != self.n_specs:
            raise ValueError("precursor_mz must have length n_specs.")

        n_global = self.peaks_mz.size
        if self.peaks_int.size != n_global or self.peaks_spec_idx.size != n_global:
            raise ValueError(
                "peaks_mz, peaks_int, and peaks_spec_idx must have identical lengths."
            )
        if n_global != self.spec_mz.size:
            raise ValueError(
                "Global product view and spectrum-major product view must contain "
                "the same number of peaks."
            )
        if n_global > 1 and np.any(self.peaks_mz[:-1] > self.peaks_mz[1:]):
            raise ValueError("peaks_mz must be globally sorted in ascending order.")
        if self.peaks_spec_idx.size:
            min_idx = int(self.peaks_spec_idx.min())
            max_idx = int(self.peaks_spec_idx.max())
            if min_idx < 0 or max_idx >= self.n_specs:
                raise ValueError("peaks_spec_idx contains out-of-range spectrum ids.")

        if self.spec_l2 is not None:
            if self.spec_l2.ndim != 1 or self.spec_l2.size != self.n_specs:
                raise ValueError("spec_l2 must have length n_specs.")

        neutral_arrays = (
            self.nl_mz,
            self.nl_int,
            self.nl_spec_idx,
            self.nl_product_idx,
        )
        have_any_nl = any(value is not None for value in neutral_arrays)
        have_all_nl = all(value is not None for value in neutral_arrays)
        if have_any_nl and not have_all_nl:
            raise ValueError(
                "Neutral-loss index arrays must either all be present or all be None."
            )
        if have_all_nl:
            n_nl = self.nl_mz.size
            if not (
                self.nl_int.size == n_nl
                and self.nl_spec_idx.size == n_nl
                and self.nl_product_idx.size == n_nl
            ):
                raise ValueError("Neutral-loss index arrays must have identical lengths.")
            if n_nl > 1 and np.any(self.nl_mz[:-1] > self.nl_mz[1:]):
                raise ValueError("nl_mz must be globally sorted in ascending order.")
            if self.nl_spec_idx.size:
                min_idx = int(self.nl_spec_idx.min())
                max_idx = int(self.nl_spec_idx.max())
                if min_idx < 0 or max_idx >= self.n_specs:
                    raise ValueError("nl_spec_idx contains out-of-range spectrum ids.")
            if self.nl_product_idx.size:
                min_pos = int(self.nl_product_idx.min())
                max_pos = int(self.nl_product_idx.max())
                if min_pos < 0 or max_pos >= n_global:
                    raise ValueError("nl_product_idx contains out-of-range product positions.")

    def __repr__(self) -> str:
        return (
            f"FlashIndex(n_specs={self.n_specs}, dtype={self.dtype}, "
            f"n_peaks={self.peaks_mz.size}, "
            f"neutral_loss={self.has_neutral_loss_index}, "
            f"l2_norms={self.has_l2_norms})"
        )
