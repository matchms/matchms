"""SpectraCollection-native helpers for Flash-based similarities.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass
class _PreparedSpectra:
    """Packed, spectrum-major representation after Flash preprocessing.

    Parameters
    ----------
    n_specs
        Number of spectra.
    spec_offsets
        CSR-like offsets into ``spec_mz`` and ``spec_int``. Spectrum ``i`` uses
        ``[spec_offsets[i]:spec_offsets[i + 1])``.
    spec_mz
        Preprocessed fragment m/z values concatenated spectrum by spectrum.
    spec_int
        Preprocessed fragment intensities aligned with ``spec_mz``.
    precursor_mz
        Precursor m/z per spectrum. Missing values are represented by ``NaN``.
    spec_l2
        Optional L2 norm per spectrum, used by cosine scoring.
    dtype
        Floating-point dtype of the packed arrays.
    """

    n_specs: int
    spec_offsets: np.ndarray
    spec_mz: np.ndarray
    spec_int: np.ndarray
    precursor_mz: np.ndarray
    spec_l2: np.ndarray | None
    dtype: np.dtype


class _LibraryIndex:
    """Compact library views used by the Flash row workers.

    The index stores the preprocessed library in two complementary layouts:

    - spectrum-major arrays (``spec_offsets``, ``spec_mz``, ``spec_int``), used
      for exact one-to-one pair scans;
    - globally m/z-sorted arrays (``peaks_mz``, ``peaks_int``,
      ``peaks_spec_idx``), used for fast candidate discovery.

    Optional neutral-loss arrays are built for ``neutral_loss`` and ``hybrid``
    matching modes.

    Attributes
    ----------
    n_specs : int
        Number of spectra in the library.
    peaks_mz : np.ndarray[dtype]
        Product (fragment) m/z values for all spectra, globally sorted ascending.
    peaks_int : np.ndarray[dtype]
        Intensities aligned 1:1 with ``peaks_mz``.
    peaks_spec_idx : np.ndarray[int32 or int64]
        Source spectrum id for each entry in ``peaks_mz``.
    spec_offsets : np.ndarray[int64]
        Prefix offsets into spectrum-major arrays. Slice for spectrum ``i`` is
        ``[spec_offsets[i]:spec_offsets[i + 1])``.
        Invariants: ``len(spec_offsets) == n_specs + 1``, ``spec_offsets[0] == 0``,
        ``spec_offsets[-1] == total number of peaks``.
    spec_mz : np.ndarray[dtype]
        Product m/z values concatenated spectrum-by-spectrum (not globally sorted).
    spec_int : np.ndarray[dtype]
        Intensities aligned 1:1 with ``spec_mz``.
    nl_mz : np.ndarray[dtype] or None
        Neutral-loss m/z values, globally sorted ascending (only when requested).
    nl_int : np.ndarray[dtype] or None
        Intensities aligned 1:1 with ``nl_mz``.
    nl_spec_idx : np.ndarray[int32 or int64] or None
        Source spectrum id for each entry in ``nl_mz``.
    nl_product_idx : np.ndarray[int64] or None
        For each NL entry, index back into global product arrays
        (``peaks_mz`` / ``peaks_int``). Used for hybrid de-duplication rules.
    spec_l2 : np.ndarray[dtype] or None
        Optional per-spectrum L2 norm of product intensities (cosine paths).
    precursor_mz : np.ndarray[dtype]
        Precursor m/z per spectrum. Unknown values are stored as ``NaN``.
    dtype : np.dtype
        Floating-point dtype used for all float arrays.
    """
    def __init__(self, dtype: np.dtype = np.float32):
        self.n_specs = 0
        self.peaks_mz = None
        self.peaks_int = None
        self.peaks_spec_idx = None
        self.nl_mz = None
        self.nl_int = None
        self.nl_spec_idx = None
        self.nl_product_idx = None
        self.spec_offsets = None
        self.spec_mz = None
        self.spec_int = None
        self.spec_l2 = None
        self.precursor_mz = None
        self.dtype = np.dtype(dtype)


def _extract_precursor_mz(collection, dtype: np.dtype) -> np.ndarray:
    """Return precursor m/z metadata as a dense floating-point array."""
    n_specs = len(collection)
    precursor_mz = np.full(n_specs, np.nan, dtype=dtype)

    metadata = collection.metadata
    if "precursor_mz" not in metadata.columns:
        return precursor_mz

    series = metadata["precursor_mz"]
    try:
        values = series.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        values = np.empty(n_specs, dtype=np.float64)
        for i, value in enumerate(series):
            try:
                values[i] = float(value)
            except (TypeError, ValueError):
                values[i] = np.nan

    precursor_mz[:] = np.asarray(values, dtype=dtype)
    return precursor_mz


def _row_ids_from_offsets(offsets: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Expand CSR-like row offsets to one row id per stored peak."""
    counts = np.diff(offsets)
    if int(offsets[-1]) > (2**31 - 1):
        dtype = np.int64
    return np.repeat(np.arange(counts.size, dtype=dtype), counts)


def _rebuild_offsets(spec_idx: np.ndarray, n_specs: int) -> np.ndarray:
    """Build spectrum-major offsets from ordered spectrum ids."""
    counts = np.bincount(spec_idx, minlength=n_specs).astype(np.int64, copy=False)
    offsets = np.empty(n_specs + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def _entropy_weight_packed(
    intensities: np.ndarray,
    spec_idx: np.ndarray,
    n_specs: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Apply Li/Fiehn entropy weighting independently to all packed spectra."""
    if intensities.size == 0:
        return intensities.astype(dtype, copy=False)

    totals = np.bincount(
        spec_idx,
        weights=intensities.astype(np.float64, copy=False),
        minlength=n_specs,
    )

    per_peak_totals = totals[spec_idx]
    p = np.zeros(intensities.shape[0], dtype=dtype)
    valid_total = per_peak_totals > 0.0
    p[valid_total] = (
        intensities[valid_total] / per_peak_totals[valid_total]
    ).astype(dtype, copy=False)

    entropy_terms = np.zeros(p.shape[0], dtype=np.float64)
    positive = p > 0.0
    entropy_terms[positive] = -(
        p[positive].astype(np.float64, copy=False)
        * np.log(p[positive]).astype(np.float64, copy=False)
    )
    entropy = np.bincount(
        spec_idx,
        weights=entropy_terms,
        minlength=n_specs,
    )

    weight = np.where(entropy >= 3.0, 1.0, 0.25 + 0.25 * entropy)
    return np.power(intensities, weight[spec_idx]).astype(dtype, copy=False)


@njit(cache=True, nogil=True)
def _merge_packed_rows_numba(
    mz: np.ndarray,
    intensities: np.ndarray,
    offsets: np.ndarray,
    max_delta_da: float,
):
    """Merge nearby peaks row-wise using the current Flash centroid rule."""
    n_specs = offsets.shape[0] - 1
    out_mz = np.empty_like(mz)
    out_int = np.empty_like(intensities)
    out_offsets = np.empty(n_specs + 1, dtype=np.int64)
    out_offsets[0] = 0

    write = 0
    for row in range(n_specs):
        start = int(offsets[row])
        end = int(offsets[row + 1])

        if start >= end:
            out_offsets[row + 1] = write
            continue

        current_mz = mz[start]
        current_int = intensities[start]

        for k in range(start + 1, end):
            if (mz[k] - current_mz) <= max_delta_da:
                total = current_int + intensities[k]
                if total > 0.0:
                    current_mz = (
                        current_mz * current_int + mz[k] * intensities[k]
                    ) / total
                    current_int = total
                else:
                    current_mz = mz[k]
                    current_int = 0.0
            else:
                out_mz[write] = current_mz
                out_int[write] = current_int
                write += 1
                current_mz = mz[k]
                current_int = intensities[k]

        out_mz[write] = current_mz
        out_int[write] = current_int
        write += 1
        out_offsets[row + 1] = write

    return out_mz[:write], out_int[:write], out_offsets


@njit(cache=True, nogil=True)
def _compute_l2_by_row_numba(
    intensities: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Return one float64 L2 norm per packed spectrum without Python row loops."""
    n_specs = offsets.shape[0] - 1
    out = np.zeros(n_specs, dtype=np.float64)

    for row in range(n_specs):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        total = 0.0
        for k in range(start, end):
            value = float(intensities[k])
            total += value * value
        out[row] = np.sqrt(total)

    return out


def _compute_l2_by_row(
    intensities: np.ndarray,
    offsets: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Return one L2 norm per packed spectrum.

    The row loop is compiled with Numba to avoid Python-level loops over millions
    of spectra becoming a noticeable part of large Cosine index construction.
    """
    return _compute_l2_by_row_numba(
        intensities,
        offsets,
    ).astype(dtype, copy=False)


def _prepare_collection(
    collection,
    *,
    intensity_power: float = 1.0,
    remove_precursor: bool = True,
    offset_to_precursor: float = -1.6,
    noise_cutoff: float | None = 0.01,
    normalize_to_half: bool = False,
    merge_within_da: float = 0.0,
    weighing_type: str = "cosine",
    compute_l2_norm: bool = False,
    dtype: np.dtype = np.float64,
) -> _PreparedSpectra:
    """Prepare an entire SpectraCollection directly from its CSR fragment store.

    The preprocessing order intentionally mirrors ``flash_utils._clean_and_weight``:

    1. remove peaks above ``precursor_mz + offset_to_precursor``;
    2. remove peaks below the row-wise relative noise cutoff;
    3. apply entropy weighting or cosine intensity power;
    4. optionally merge nearby peaks by intensity-weighted centroid;
    5. optionally normalize each spectrum to total intensity 0.5.

    No ``Spectrum`` objects are reconstructed.
    """
    dtype = np.dtype(dtype)
    if weighing_type not in ("cosine", "entropy"):
        raise ValueError(f"Score type '{weighing_type}' not recognized.")

    fragments = collection.fragments
    if not hasattr(fragments, "array") or not hasattr(fragments, "bin_to_mz"):
        raise TypeError(
            "SpectraCollection-native Flash scoring currently requires a fragment "
            "backend exposing 'array' and 'bin_to_mz' (CSRFragmentCollection)."
        )

    csr = fragments.array
    if not csr.has_sorted_indices:
        csr = csr.sorted_indices()

    n_specs = len(collection)
    precursor_mz = _extract_precursor_mz(collection, dtype)

    if csr.nnz == 0:
        offsets = np.zeros(n_specs + 1, dtype=np.int64)
        empty = np.zeros(0, dtype=dtype)
        spec_l2 = np.zeros(n_specs, dtype=dtype) if compute_l2_norm else None
        return _PreparedSpectra(
            n_specs=n_specs,
            spec_offsets=offsets,
            spec_mz=empty,
            spec_int=empty.copy(),
            precursor_mz=precursor_mz,
            spec_l2=spec_l2,
            dtype=dtype,
        )

    raw_offsets = np.asarray(csr.indptr, dtype=np.int64)
    spec_idx = _row_ids_from_offsets(raw_offsets)
    mz = np.asarray(fragments.bin_to_mz(csr.indices), dtype=dtype)
    intensities = np.asarray(csr.data, dtype=dtype)

    keep = np.ones(mz.shape[0], dtype=bool)

    # 1) Precursor-region removal. Missing precursor values leave the row unchanged.
    if remove_precursor:
        peak_pmz = precursor_mz[spec_idx]
        have_pmz = np.isfinite(peak_pmz)
        keep &= (~have_pmz) | (mz <= peak_pmz + offset_to_precursor)

    if not np.all(keep):
        mz = mz[keep]
        intensities = intensities[keep]
        spec_idx = spec_idx[keep]

    if mz.size == 0:
        offsets = np.zeros(n_specs + 1, dtype=np.int64)
        spec_l2 = np.zeros(n_specs, dtype=dtype) if compute_l2_norm else None
        return _PreparedSpectra(
            n_specs=n_specs,
            spec_offsets=offsets,
            spec_mz=np.zeros(0, dtype=dtype),
            spec_int=np.zeros(0, dtype=dtype),
            precursor_mz=precursor_mz,
            spec_l2=spec_l2,
            dtype=dtype,
        )

    # 2) Relative noise filtering, after precursor removal
    if noise_cutoff and noise_cutoff > 0.0:
        row_max = np.zeros(n_specs, dtype=dtype)
        np.maximum.at(row_max, spec_idx, intensities)
        keep = intensities >= row_max[spec_idx] * float(noise_cutoff)
        mz = mz[keep]
        intensities = intensities[keep]
        spec_idx = spec_idx[keep]

    if mz.size == 0:
        offsets = np.zeros(n_specs + 1, dtype=np.int64)
        spec_l2 = np.zeros(n_specs, dtype=dtype) if compute_l2_norm else None
        return _PreparedSpectra(
            n_specs=n_specs,
            spec_offsets=offsets,
            spec_mz=np.zeros(0, dtype=dtype),
            spec_int=np.zeros(0, dtype=dtype),
            precursor_mz=precursor_mz,
            spec_l2=spec_l2,
            dtype=dtype,
        )

    # 3) Score-specific intensity weighting.
    if weighing_type == "entropy":
        intensities = _entropy_weight_packed(
            intensities,
            spec_idx,
            n_specs,
            dtype,
        )
    elif intensity_power != 1.0:
        intensities = np.power(intensities, intensity_power).astype(dtype, copy=False)

    offsets = _rebuild_offsets(spec_idx, n_specs)

    # 4) Optional row-local peak merging. This is the only inherently segmented
    # preprocessing step; it still works directly on packed arrays without Spectrum objects.
    if merge_within_da and merge_within_da > 0.0 and mz.size > 1:
        mz, intensities, offsets = _merge_packed_rows_numba(
            mz,
            intensities,
            offsets,
            float(merge_within_da),
        )

    # 5) Optional row-wise normalization to total intensity 0.5.
    if normalize_to_half and intensities.size > 0:
        spec_idx = _row_ids_from_offsets(offsets)
        row_sum = np.bincount(
            spec_idx,
            weights=intensities.astype(np.float64, copy=False),
            minlength=n_specs,
        )
        scale = np.zeros(n_specs, dtype=np.float64)
        nonzero = row_sum > 0.0
        scale[nonzero] = 0.5 / row_sum[nonzero]
        intensities = (
            intensities * scale[spec_idx]
        ).astype(dtype, copy=False)

    spec_l2 = (
        _compute_l2_by_row(intensities, offsets, dtype)
        if compute_l2_norm
        else None
    )

    return _PreparedSpectra(
        n_specs=n_specs,
        spec_offsets=np.asarray(offsets, dtype=np.int64),
        spec_mz=np.asarray(mz, dtype=dtype),
        spec_int=np.asarray(intensities, dtype=dtype),
        precursor_mz=precursor_mz,
        spec_l2=spec_l2,
        dtype=dtype,
    )


def _build_library_index_from_prepared(
    prepared: _PreparedSpectra,
    *,
    compute_neutral_loss: bool = False,
    compute_l2_norm: bool = False,
) -> _LibraryIndex:
    """Build the Flash search index directly from packed prepared spectra."""
    idx = _LibraryIndex(prepared.dtype)
    idx.n_specs = prepared.n_specs
    idx.precursor_mz = prepared.precursor_mz
    idx.spec_offsets = prepared.spec_offsets
    idx.spec_mz = prepared.spec_mz
    idx.spec_int = prepared.spec_int

    n_peaks = prepared.spec_mz.shape[0]
    int_dtype = np.int64 if n_peaks > (2**31 - 1) else np.int32

    if compute_l2_norm:
        if prepared.spec_l2 is None:
            idx.spec_l2 = _compute_l2_by_row(
                prepared.spec_int,
                prepared.spec_offsets,
                prepared.dtype,
            )
        else:
            idx.spec_l2 = prepared.spec_l2

    if n_peaks == 0:
        idx.peaks_mz = np.zeros(0, dtype=prepared.dtype)
        idx.peaks_int = np.zeros(0, dtype=prepared.dtype)
        idx.peaks_spec_idx = np.zeros(0, dtype=int_dtype)
        if compute_neutral_loss:
            idx.nl_mz = np.zeros(0, dtype=prepared.dtype)
            idx.nl_int = np.zeros(0, dtype=prepared.dtype)
            idx.nl_spec_idx = np.zeros(0, dtype=int_dtype)
            idx.nl_product_idx = np.zeros(0, dtype=np.int64)
        return idx

    spec_flat = _row_ids_from_offsets(prepared.spec_offsets, dtype=int_dtype)

    # Global m/z-sorted product view for search windows.
    order = np.argsort(prepared.spec_mz)
    idx.peaks_mz = prepared.spec_mz[order]
    idx.peaks_int = prepared.spec_int[order]
    idx.peaks_spec_idx = spec_flat[order]

    if compute_neutral_loss:
        # Map spectrum-major peak positions to positions in the globally sorted
        # product arrays. This mapping can be very large, so allocate it only
        # when neutral-loss/hybrid indexing actually needs it.
        product_pos = np.empty(n_peaks, dtype=np.int64)
        product_pos[order] = np.arange(n_peaks, dtype=np.int64)

        pmz_per_peak = prepared.precursor_mz[spec_flat]
        have_pmz = np.isfinite(pmz_per_peak)
        src_idx = np.nonzero(have_pmz)[0]

        if src_idx.size == 0:
            idx.nl_mz = np.zeros(0, dtype=prepared.dtype)
            idx.nl_int = np.zeros(0, dtype=prepared.dtype)
            idx.nl_spec_idx = np.zeros(0, dtype=int_dtype)
            idx.nl_product_idx = np.zeros(0, dtype=np.int64)
        else:
            nl_mz = (
                pmz_per_peak[src_idx] - prepared.spec_mz[src_idx]
            ).astype(prepared.dtype, copy=False)
            nl_int = prepared.spec_int[src_idx]
            nl_spec = spec_flat[src_idx]
            nl_prod = product_pos[src_idx]

            order_nl = np.argsort(nl_mz)
            idx.nl_mz = nl_mz[order_nl]
            idx.nl_int = nl_int[order_nl]
            idx.nl_spec_idx = nl_spec[order_nl]
            idx.nl_product_idx = nl_prod[order_nl]

    return idx


# -----------------------------------------------------------------------------
# Spectrum-oriented helpers retained for pair-level compatibility and direct
# comparison with the old spectrum based implementation.
# -----------------------------------------------------------------------------

def _entropy_weight(intensities: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Apply entropy-based weighting to one intensity array."""
    total = float(intensities.sum(dtype=np.float64))
    if total <= 0.0:
        return intensities.astype(dtype, copy=False)

    p = intensities / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.zeros_like(p, dtype=dtype)
        mask = p > 0
        logp[mask] = np.log(p[mask]).astype(dtype, copy=False)

    entropy = float((-p * logp).sum(dtype=np.float64))
    weight = 1.0 if entropy >= 3.0 else (0.25 + 0.25 * entropy)
    return np.power(intensities, weight).astype(dtype, copy=False)


def _clean_and_weight(
    peaks: np.ndarray,
    precursor_mz: float | None,
    remove_precursor: bool,
    offset_to_precursor: float,
    noise_cutoff: float,
    normalize_to_half: bool,
    merge_within_da: float,
    weighing_type: str,
    intensity_power: float = 1.0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Apply the existing Flash preprocessing rules to a single peak array."""
    dtype = np.dtype(dtype)
    if peaks.size == 0:
        return np.empty((0, 2), dtype=dtype)

    mz = np.asarray(peaks[:, 0], dtype=dtype)
    intensities = np.asarray(peaks[:, 1], dtype=dtype)

    if remove_precursor and precursor_mz is not None:
        keep = mz <= float(precursor_mz) + float(offset_to_precursor)
        mz = mz[keep]
        intensities = intensities[keep]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    if noise_cutoff and noise_cutoff > 0.0:
        threshold = intensities.max() * float(noise_cutoff)
        keep = intensities >= threshold
        mz = mz[keep]
        intensities = intensities[keep]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    if weighing_type == "entropy":
        intensities = _entropy_weight(intensities, dtype)
    elif weighing_type == "cosine":
        if intensity_power != 1.0:
            intensities = np.power(intensities, intensity_power).astype(dtype, copy=False)
    else:
        raise ValueError(f"Score type '{weighing_type}' not recognized.")

    if merge_within_da and merge_within_da > 0.0 and mz.size > 1:
        offsets = np.array([0, mz.size], dtype=np.int64)
        mz, intensities, _ = _merge_packed_rows_numba(
            mz,
            intensities,
            offsets,
            float(merge_within_da),
        )

    if normalize_to_half:
        total = intensities.sum(dtype=np.float64)
        if total > 0.0:
            intensities = (intensities * (0.5 / total)).astype(dtype, copy=False)

    return np.column_stack((mz, intensities))
