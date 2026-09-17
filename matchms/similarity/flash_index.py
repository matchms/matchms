"""Persistent peak indexes for cosine and spectral entropy library search.

The globally sorted view supports tolerance-window lookups, while the
spectrum-major view retains peak identities and per-spectrum normalization.
The versioned, non-pickle archive format stores preprocessing settings alongside
both views. Search tolerances and output selection are independent of the index.
"""
from __future__ import annotations
import json
import operator
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
import numpy as np
from ._flash_prepared import readonly


_FLASH_INDEX_FORMAT = "matchms.flash_index"
_FLASH_INDEX_VERSION = 3
_SUPPORTED_VERSIONS = (2, 3)
_REQUIRED_ARRAYS = (
    "peaks_mz", "peaks_int", "peaks_spec_idx", "spec_offsets",
    "spec_mz", "spec_int", "precursor_mz",
)
_OPTIONAL_ARRAYS = (
    "spec_l2", "nl_mz", "nl_int", "nl_spec_idx", "nl_product_idx",
    "peaks_pid", "peaks_xlog2", "nl_xlog2",
)
_NL_ARRAYS = ("nl_mz", "nl_int", "nl_spec_idx", "nl_product_idx")


@dataclass(frozen=True)
class _EntropyKernelData:
    """Read-only peak views, entropy terms, and physical IDs for the row kernel."""
    n_specs: int
    n_peaks: int
    precursor_mz: np.ndarray
    fragment: tuple
    neutral_loss: tuple
    dtype: np.dtype


@dataclass(frozen=True)
class _CosineKernelData:
    """Read-only argument bundle with one-time dtype conversions."""
    n_specs: int
    n_peaks: int
    peaks_mz: np.ndarray
    peaks_int: np.ndarray
    peaks_spec_idx: np.ndarray
    nl_mz: np.ndarray
    nl_spec_idx: np.ndarray
    nl_product_idx: np.ndarray
    precursor_mz: np.ndarray
    spec_l2: np.ndarray
    dtype: np.dtype


def config_from_settings(settings: tuple) -> dict[str, Any]:
    """Translate a preparation-settings tuple into the saved configuration."""
    kind, power, remove, offset, noise, half, merge, dtype = settings
    return {
        "weighing_type": kind,
        "intensity_power": float(power),
        "remove_precursor": bool(remove),
        "offset_to_precursor": float(offset),
        "noise_cutoff": float(noise or 0.0),
        "normalize_to_half": bool(half),
        "merge_within": float(merge),
        "dtype": np.dtype(dtype).str,
    }


def _xlog2_array(values: np.ndarray) -> np.ndarray:
    """Compute float64 x-log-x terms, assigning zero to zero intensities."""
    value = np.asarray(values, dtype=np.float64)
    result = np.zeros(value.size, dtype=np.float64)
    positive = value > 0
    result[positive] = value[positive] * np.log2(value[positive])
    return result


def _ordered_posting(
    coordinates, values, xlog, owners, pid, *, reverse_ties: bool = False,
) -> tuple:
    """Keep existing order unless coordinate or physical-ID ordering needs repair."""
    tie_key = -pid if reverse_ties else pid
    needs_sort = coordinates.size > 1 and np.any(
        (coordinates[1:] < coordinates[:-1])
        | ((coordinates[1:] == coordinates[:-1]) & (tie_key[1:] < tie_key[:-1]))
    )
    arrays = (coordinates, values, xlog, owners, pid)
    if needs_sort:
        order = np.lexsort((tie_key, coordinates))
        arrays = tuple(value[order] for value in arrays)
    return tuple(readonly(value) for value in arrays)


@dataclass
class FlashIndex:
    """Library peak index with persistent preparation settings.

    Attributes
    ----------
    n_specs
        Number of library spectra, including empty rows.
    dtype
        Floating-point dtype of the product coordinates and intensities.
    peaks_mz, peaks_int, peaks_spec_idx
        Globally m/z-sorted product coordinates, intensities, and spectrum IDs.
    spec_offsets, spec_mz, spec_int
        Spectrum-major peak storage. The peaks of spectrum i occupy
        ``spec_offsets[i]:spec_offsets[i + 1]``.
    precursor_mz
        Precursor coordinates; unknown values are NaN.
    spec_l2
        Optional per-spectrum L2 norms required for cosine scoring.
    nl_mz, nl_int, nl_spec_idx, nl_product_idx
        Optional neutral-loss index. ``nl_product_idx`` addresses the globally
        sorted product arrays, not the spectrum-major arrays.
    peaks_pid
        Optional mapping from each global product position to its physical
        position in ``spec_mz`` and ``spec_int``. Entropy uses physical IDs to
        enforce one-to-one matching across fragment and neutral-loss passes.
    peaks_xlog2, nl_xlog2
        Optional cached intensity-logarithm terms for entropy accumulation.
    config
        Preprocessing configuration. Matching tolerance, ppm matching, output
        fields, and identity gates are not part of this configuration.
    metadata
        Descriptive information such as source fragment backend and m/z precision.

    Notes
    -----
    Treat index arrays as read-only after construction. ``from_library`` shares
    its input arrays; kernel views and dtype conversions are cached on this
    instance. After intentional edits, call ``validate`` to check consistency and
    discard runtime views. Derived arrays must also agree with the edited data.

    Loss-capable indexes can serve fragment, loss, or hybrid searches. A
    fragment-only index cannot supply loss matches and is not silently rebuilt.
    Public format versions 2 and 3 are readable. Missing derived arrays in older
    indexes are recovered once without reprocessing the source spectra.
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
    # Optional derived data follow existing positional constructor arguments so
    # callers and version-2 archives need not supply them.
    peaks_pid: np.ndarray | None = None
    peaks_xlog2: np.ndarray | None = None
    nl_xlog2: np.ndarray | None = None
    _runtime_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.n_specs, (bool, np.bool_)):
            raise TypeError("FlashIndex.n_specs must be an integer.")
        self.n_specs = operator.index(self.n_specs)
        self.dtype = np.dtype(self.dtype)
        self.config = dict(self.config or {})
        self.metadata = dict(self.metadata or {})
        # Never reuse a caller's cache, including dataclass copies with edits.
        self._runtime_cache = {}
        self._validate_arrays()

    @classmethod
    def from_library(
        cls, library, *, config: dict, metadata: dict | None = None,
    ) -> Self:
        """Wrap the native library arrays without copying them."""
        return cls(
            n_specs=library.n_specs, dtype=library.dtype,
            **{name: getattr(library, name) for name in _REQUIRED_ARRAYS},
            **{name: getattr(library, name, None) for name in _OPTIONAL_ARRAYS},
            config=config, metadata=metadata or {},
        )

    @property
    def n_spectra(self) -> int:
        """Number of indexed library spectra."""
        return self.n_specs

    @property
    def n_peaks(self) -> int:
        """Total number of indexed product peaks."""
        return self.spec_mz.size

    @property
    def has_neutral_loss_index(self) -> bool:
        """Whether all neutral-loss index arrays are available."""
        return self.nl_mz is not None

    @property
    def has_l2_norms(self) -> bool:
        """Whether per-spectrum cosine normalization factors are available."""
        return self.spec_l2 is not None

    def validate(self) -> None:
        """Validate after explicit edits; normal search never rescans the index."""
        self._runtime_cache = {}
        self._validate_arrays()

    def _validate_arrays(self) -> None:
        """Check dimensions, dtypes and bounds before arrays can reach Numba."""
        for name in _REQUIRED_ARRAYS + _OPTIONAL_ARRAYS:
            value = getattr(self, name)
            if value is None and name in _OPTIONAL_ARRAYS:
                continue
            if not isinstance(value, np.ndarray):
                raise TypeError(f"FlashIndex.{name} must be a NumPy array.")
            if value.ndim != 1:
                raise ValueError(f"FlashIndex.{name} must be a 1D array.")
        if self.n_specs < 0:
            raise ValueError("FlashIndex.n_specs must be >= 0.")
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("FlashIndex.dtype must be float32 or float64.")
        if self.spec_offsets.size != self.n_specs + 1:
            raise ValueError("spec_offsets must have length n_specs + 1.")
        if self.spec_offsets.dtype.kind not in "iu":
            raise ValueError("spec_offsets must contain integers.")
        if int(self.spec_offsets[0]) != 0:
            raise ValueError("spec_offsets must start at 0.")
        if np.any(self.spec_offsets[1:] < self.spec_offsets[:-1]):
            raise ValueError("spec_offsets must be monotonically non-decreasing.")
        if self.spec_mz.size != self.spec_int.size:
            raise ValueError("spec_mz and spec_int must have identical lengths.")
        if int(self.spec_offsets[-1]) != self.spec_mz.size:
            raise ValueError("spec_offsets[-1] must equal the number of spectrum-major peaks.")
        if self.precursor_mz.size != self.n_specs:
            raise ValueError("precursor_mz must have length n_specs.")
        n_peaks = self.peaks_mz.size
        if self.peaks_int.size != n_peaks or self.peaks_spec_idx.size != n_peaks:
            raise ValueError("peaks_mz, peaks_int, and peaks_spec_idx must have identical lengths.")
        if n_peaks != self.spec_mz.size:
            raise ValueError(
                "Global product view and spectrum-major product view must contain "
                "the same number of peaks."
            )
        if n_peaks > 1 and np.any(self.peaks_mz[:-1] > self.peaks_mz[1:]):
            raise ValueError("peaks_mz must be globally sorted in ascending order.")
        self._ids("peaks_spec_idx", self.peaks_spec_idx, self.n_specs, "spectrum ids")
        for name in ("peaks_mz", "peaks_int", "spec_mz", "spec_int"):
            value = getattr(self, name)
            if value.dtype != self.dtype:
                raise ValueError(f"{name} must have the declared index dtype {self.dtype}.")
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must contain finite, nonnegative values.")
        if self.precursor_mz.dtype.kind != "f" or np.any(np.isinf(self.precursor_mz)):
            raise ValueError("precursor_mz must be floating-point, with NaN for missing values.")
        # Sorted within rows, but not necessarily between rows (empty rows too).
        if n_peaks > 1:
            backwards = self.spec_mz[1:] < self.spec_mz[:-1]
            row_boundaries = self.spec_offsets[1:-1]
            row_boundaries = row_boundaries[(row_boundaries > 0) & (row_boundaries < n_peaks)]
            backwards[row_boundaries - 1] = False
            if np.any(backwards):
                raise ValueError("spec_mz must be sorted within each spectrum.")
        if self.spec_l2 is not None:
            if self.spec_l2.size != self.n_specs:
                raise ValueError("spec_l2 must have length n_specs.")
            if (self.spec_l2.dtype.kind != "f" or not np.all(np.isfinite(self.spec_l2))
                    or np.any(self.spec_l2 < 0)):
                raise ValueError("spec_l2 must contain finite, nonnegative floating-point norms.")
        have_nl = [getattr(self, name) is not None for name in _NL_ARRAYS]
        if any(have_nl) and not all(have_nl):
            raise ValueError("Neutral-loss index arrays must either all be present or all be None.")
        if all(have_nl):
            n_losses = self.nl_mz.size
            if not all(getattr(self, name).size == n_losses for name in _NL_ARRAYS):
                raise ValueError("Neutral-loss index arrays must have identical lengths.")
            if self.nl_mz.dtype.kind != "f" or not np.all(np.isfinite(self.nl_mz)):
                raise ValueError("nl_mz must contain finite floating-point coordinates.")
            if n_losses > 1 and np.any(self.nl_mz[:-1] > self.nl_mz[1:]):
                raise ValueError("nl_mz must be globally sorted in ascending order.")
            self._ids("nl_spec_idx", self.nl_spec_idx, self.n_specs, "spectrum ids")
            self._ids("nl_product_idx", self.nl_product_idx, n_peaks, "product positions")
            if self.nl_int.dtype != self.dtype or not np.all(np.isfinite(self.nl_int)) or np.any(self.nl_int < 0):
                raise ValueError("nl_int must contain finite, nonnegative intensities of the index dtype.")
            if not np.array_equal(self.nl_spec_idx, self.peaks_spec_idx[self.nl_product_idx]):
                raise ValueError("nl_product_idx maps a neutral loss to the wrong spectrum.")
            if not np.array_equal(self.nl_int, self.peaks_int[self.nl_product_idx]):
                raise ValueError("nl_int disagrees with its mapped product intensities.")
            if not np.all(np.isfinite(self.precursor_mz[self.nl_spec_idx])):
                raise ValueError("Neutral-loss entries require finite precursor_mz values.")
        if self.peaks_pid is not None:
            if self.peaks_pid.size != n_peaks:
                raise ValueError("peaks_pid must have one entry per global product peak.")
            self._ids("peaks_pid", self.peaks_pid, n_peaks, "physical peak positions")
            if n_peaks and np.any(np.bincount(self.peaks_pid.astype(np.int64), minlength=n_peaks) != 1):
                raise ValueError("peaks_pid must be a permutation of physical peak positions.")
            if (not np.array_equal(self.peaks_mz, self.spec_mz[self.peaks_pid])
                    or not np.array_equal(self.peaks_int, self.spec_int[self.peaks_pid])):
                raise ValueError("peaks_pid does not map the global and spectrum-major views consistently.")
            starts = self.spec_offsets[self.peaks_spec_idx]
            stops = self.spec_offsets[self.peaks_spec_idx + 1]
            if np.any(self.peaks_pid < starts) or np.any(self.peaks_pid >= stops):
                raise ValueError("peaks_pid maps a peak to the wrong spectrum.")
        for name, values in (("peaks_xlog2", self.peaks_int), ("nl_xlog2", self.nl_int)):
            cached = getattr(self, name)
            if cached is None:
                continue
            if values is None or cached.shape != values.shape or cached.dtype != np.float64:
                raise ValueError(f"{name} must be float64 and aligned to its intensity array.")
            if not np.all(np.isfinite(cached)) or not np.allclose(cached, _xlog2_array(values), rtol=1e-13, atol=1e-15):
                raise ValueError(f"{name} disagrees with its intensity array.")

    @staticmethod
    def _ids(name: str, array: np.ndarray, upper: int, description: str) -> None:
        """Validate an integer ID vector against its exclusive upper bound."""
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer {description}.")
        if array.size and (np.any(array < 0) or np.any(array >= upper)):
            raise ValueError(f"{name} contains out-of-range {description}.")

    def _physical_peak_ids(self) -> np.ndarray:
        """Recover version-2 mappings without access to original spectra.

        Sorting records by (spectrum, m/z, intensity) disambiguates equal m/z
        peaks with different intensities. Exactly duplicate peaks are assigned
        deterministically. This migration is only needed once for older indices.
        """
        if self.peaks_pid is not None:
            return self.peaks_pid
        owners = np.repeat(np.arange(self.n_specs, dtype=np.int64), np.diff(self.spec_offsets))
        local_order = np.lexsort((self.spec_int, self.spec_mz, owners))
        global_order = np.lexsort((self.peaks_int, self.peaks_mz, self.peaks_spec_idx))
        if (not np.array_equal(owners[local_order], self.peaks_spec_idx[global_order])
                or not np.array_equal(self.spec_mz[local_order], self.peaks_mz[global_order])
                or not np.array_equal(self.spec_int[local_order], self.peaks_int[global_order])):
            raise ValueError("Global and spectrum-major peak views are inconsistent.")
        physical_ids = np.empty(self.n_peaks, dtype=np.int64)
        physical_ids[global_order] = local_order
        self.peaks_pid = physical_ids
        return physical_ids

    def entropy_data(self) -> _EntropyKernelData:
        """Prepare entropy kernel views once (also works with version-2 indices)."""
        if "entropy" in self._runtime_cache:
            return self._runtime_cache["entropy"]
        pid = np.asarray(self._physical_peak_ids(), dtype=np.int64)
        if self.peaks_xlog2 is None:
            self.peaks_xlog2 = _xlog2_array(self.peaks_int)
        fragment = _ordered_posting(
            self.peaks_mz, self.peaks_int, self.peaks_xlog2,
            np.asarray(self.peaks_spec_idx, dtype=np.int64), pid,
        )
        empty = (readonly(np.empty(0, np.float64)), readonly(np.empty(0, self.dtype)),
                 readonly(np.empty(0, np.float64)), readonly(np.empty(0, np.int64)),
                 readonly(np.empty(0, np.int64)))
        neutral_loss = empty
        if self.has_neutral_loss_index:
            if self.nl_xlog2 is None:
                self.nl_xlog2 = self.peaks_xlog2[self.nl_product_idx]
            n_pid = pid[self.nl_product_idx]
            # Evaluate precursor-minus-fragment in float64, just like queries.
            # In old archives stored NL coordinates can be rounded to float32.
            losses = (self.precursor_mz[self.nl_spec_idx].astype(np.float64)
                      - self.spec_mz[n_pid].astype(np.float64))
            neutral_loss = _ordered_posting(
                losses, self.nl_int, self.nl_xlog2,
                np.asarray(self.nl_spec_idx, dtype=np.int64), n_pid, reverse_ties=True,
            )
        data = _EntropyKernelData(
            self.n_specs, self.n_peaks, readonly(self.precursor_mz, np.float64),
            fragment, neutral_loss, self.dtype,
        )
        self._runtime_cache["entropy"] = data
        return data

    def cosine_data(self) -> _CosineKernelData:
        """Cast IDs and norms once; repeated searches reuse these read-only views."""
        if "cosine" in self._runtime_cache:
            return self._runtime_cache["cosine"]
        if not self.has_l2_norms:
            raise ValueError("Cosine search requires an index containing L2 norms.")
        nl_mz = self.nl_mz if self.has_neutral_loss_index else np.empty(0, self.dtype)
        nl_ids = self.nl_spec_idx if self.has_neutral_loss_index else np.empty(0, np.int64)
        nl_pos = self.nl_product_idx if self.has_neutral_loss_index else np.empty(0, np.int64)
        data = _CosineKernelData(
            self.n_specs, self.n_peaks, readonly(self.peaks_mz), readonly(self.peaks_int),
            readonly(self.peaks_spec_idx, np.int64), readonly(nl_mz), readonly(nl_ids, np.int64),
            readonly(nl_pos, np.int64), readonly(self.precursor_mz, np.float64),
            readonly(self.spec_l2, np.float64), self.dtype,
        )
        self._runtime_cache["cosine"] = data
        return data

    def save(self, filename: str | Path, *, overwrite: bool = True) -> None:
        """Save arrays and configuration in a versioned, non-pickle NPZ archive.

        A temporary file and atomic replacement avoid leaving a partial index
        at the destination. Uncompressed storage prioritizes write/load speed.
        Set ``overwrite=False`` to refuse an existing destination. Runtime views
        are not serialized; optional derived arrays are saved when available.
        """
        self._validate_arrays()
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        meta = {
            "format": _FLASH_INDEX_FORMAT, "version": _FLASH_INDEX_VERSION,
            "n_specs": self.n_specs, "dtype": self.dtype.str,
            "config": self.config, "metadata": self.metadata,
            "optional_arrays": {name: getattr(self, name) is not None for name in _OPTIONAL_ARRAYS},
        }
        arrays = {name: getattr(self, name) for name in _REQUIRED_ARRAYS + _OPTIONAL_ARRAYS
                  if getattr(self, name) is not None}
        payload = json.dumps(meta)  # Validate JSON before opening the temporary file.
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                tmp = Path(handle.name)
                np.savez(handle, __metadata__=np.asarray(payload), **arrays)
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, filename: str | Path) -> Self:
        """Load version 2 or 3 of the public archive, with structural validation."""
        try:
            with np.load(filename, allow_pickle=False) as archive:
                meta = json.loads(str(archive["__metadata__"].item()))
                if not isinstance(meta, dict):
                    raise TypeError("Flash index metadata must be a JSON object.")
                if meta.get("format") != _FLASH_INDEX_FORMAT:
                    raise ValueError(f"Not a matchms Flash index: {meta.get('format')!r}.")
                version = meta.get("version")
                if type(version) is not int or version not in _SUPPORTED_VERSIONS:
                    raise ValueError(
                        f"Unsupported Flash index version {version!r}; "
                        f"expected one of {_SUPPORTED_VERSIONS}."
                    )
                optional = meta.get("optional_arrays", {})
                if not isinstance(optional, dict):
                    raise TypeError("Malformed optional_arrays metadata.")
                arrays = {}
                for name in _REQUIRED_ARRAYS:
                    if name not in archive:
                        raise ValueError(f"Flash index is missing required array {name!r}.")
                    arrays[name] = np.asarray(archive[name])
                for name in _OPTIONAL_ARRAYS:
                    present = optional.get(name, False)
                    if type(present) is not bool:
                        raise ValueError(f"Invalid optional-array flag for {name!r}.")
                    if present and name not in archive:
                        raise ValueError(
                            f"Flash index metadata declares array {name!r}, "
                            "but the array is missing from the file."
                        )
                    arrays[name] = np.asarray(archive[name]) if present else None
                return cls(
                    n_specs=meta["n_specs"], dtype=np.dtype(meta["dtype"]),
                    config=meta.get("config", {}), metadata=meta.get("metadata", {}), **arrays,
                )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed Flash index archive.") from exc

    def __repr__(self) -> str:
        return (
            f"FlashIndex(n_specs={self.n_specs}, dtype={self.dtype}, "
            f"n_peaks={self.n_peaks}, neutral_loss={self.has_neutral_loss_index}, "
            f"l2_norms={self.has_l2_norms})"
        )


def build_entropy_index(
    prepared, mode: str, *, config: dict | None = None, metadata: dict | None = None,
) -> FlashIndex:
    """Construct entropy index views from prepared peaks.

    Coordinate ties follow ascending physical peak IDs for fragments and
    descending IDs for losses, matching the two traversal directions.
    """
    if mode not in ("fragment", "neutral_loss", "hybrid"):
        raise ValueError("Unknown matching mode.")
    mz = prepared.spec_mz
    intensities = prepared.spec_int
    n_spectra = prepared.n_specs
    n_peaks = mz.size
    spectrum_ids = np.repeat(
        np.arange(n_spectra, dtype=np.int64), np.diff(prepared.spec_offsets),
    )
    physical_ids = np.arange(n_peaks, dtype=np.int64)
    precursors = np.asarray(prepared.precursor_mz, dtype=np.float64)
    product_order = np.argsort(mz, kind="stable")
    loss_arrays = {}
    if mode != "fragment":
        valid_ids = physical_ids[np.isfinite(precursors[spectrum_ids])][::-1]
        losses = precursors[spectrum_ids[valid_ids]] - mz[valid_ids].astype(np.float64)
        loss_order = np.argsort(losses, kind="stable")
        loss_ids = valid_ids[loss_order]
        product_positions = np.empty(n_peaks, dtype=np.int64)
        product_positions[product_order] = physical_ids
        loss_arrays = {
            "nl_mz": losses[loss_order],
            "nl_int": intensities[loss_ids],
            "nl_spec_idx": spectrum_ids[loss_ids],
            "nl_product_idx": product_positions[loss_ids],
        }
    index = FlashIndex(
        n_specs=n_spectra,
        dtype=intensities.dtype,
        peaks_mz=mz[product_order],
        peaks_int=intensities[product_order],
        peaks_spec_idx=spectrum_ids[product_order],
        spec_offsets=prepared.spec_offsets,
        spec_mz=mz,
        spec_int=intensities,
        precursor_mz=precursors,
        peaks_pid=product_order,
        config=config or {},
        metadata=metadata or {},
        **loss_arrays,
    )
    # Derive logarithms after validating the base arrays. Fresh indexes compute
    # these terms once; saved terms are checked for consistency when loaded.
    index.entropy_data()
    return index


def build_index(prepared, matching_mode: str, kind: str, *, metadata: dict | None = None) -> FlashIndex:
    """Build the coordinate views and normalization required by a similarity."""
    config = config_from_settings(prepared.settings)
    config["compute_neutral_loss"] = matching_mode != "fragment"
    if kind == "entropy":
        return build_entropy_index(prepared, matching_mode, config=config, metadata=metadata)
    if kind != "cosine":
        raise ValueError(f"Unsupported index kind: {kind!r}.")
    from .flash_utils import _build_library_index_from_prepared

    native = _build_library_index_from_prepared(
        prepared,
        compute_neutral_loss=matching_mode != "fragment",
        compute_l2_norm=True,
    )
    index = FlashIndex.from_library(native, config=config, metadata=metadata)
    index.cosine_data()
    return index
