"""Spectral entropy search for spectra with well-separated fragment peaks.

This module implements direct accumulation from an inverted fragment index.
Within-spectrum peak separation removes the need for one-to-one assignment
bookkeeping. Short matching intervals are processed by a compiled scalar loop;
long intervals use NumPy's vectorized logarithm followed by compiled scatter-add.
Neither path approximates the entropy formula or discretizes matching masses.
"""
from __future__ import annotations
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
import numpy as np
from numba import njit
from tqdm.auto import tqdm
from matchms.scores import Scores
from .base_similarity import BaseSimilarity
from .default_parameters import (
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_index import FlashIndex


# These control execution only, not matching, weighting, or score precision.
_VECTOR_MIN_HITS = 512
_VECTOR_CHUNK_SIZE = 65_536
_MAX_DIRECTORY_ENTRIES = 2_000_000
_PREPARATION_FORMAT = "matchms.entropy_search.1"
_CACHE_PREFIX = "entropy_search.1"


def _readonly(array: np.ndarray, dtype=None) -> np.ndarray:
    """Return a contiguous read-only view without changing caller-owned flags."""
    result = np.ascontiguousarray(array, dtype=dtype).view()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class _PreparedEntropySpectra:
    """Opaque, prepared query/library batch returned by ``prepare_queries``.

    Intensities are entropy-weighted and sum to 0.5 for every nonempty row.
    Obtain instances from the scorer, not by manually constructing this class.
    """

    spec_offsets: np.ndarray
    spec_mz: np.ndarray
    spec_int: np.ndarray
    precursor_mz: np.ndarray
    settings: tuple
    n_input_peaks: int
    n_filtered_peaks: int
    n_merged_peaks: int

    @property
    def n_specs(self) -> int:
        return self.spec_offsets.size - 1

    @property
    def n_peaks(self) -> int:
        return self.spec_mz.size

    def __len__(self) -> int:
        return self.n_specs

    def slice(self, start: int, stop: int) -> _PreparedEntropySpectra:
        """Select contiguous rows, sharing immutable prepared peak buffers."""
        if not 0 <= start <= stop <= self.n_specs:
            raise ValueError("Invalid prepared-spectrum row slice.")
        first, last = self.spec_offsets[start], self.spec_offsets[stop]
        return _PreparedEntropySpectra(
            _readonly(self.spec_offsets[start:stop + 1] - first),
            self.spec_mz[first:last], self.spec_int[first:last],
            self.precursor_mz[start:stop], self.settings,
            int(last - first), 0, 0,
        )


@dataclass(frozen=True)
class _SearchView:
    """Compact arrays and optional directory cached on an ordinary FlashIndex."""

    mz: np.ndarray
    intensity: np.ndarray
    terms: np.ndarray
    spectrum_ids: np.ndarray
    starts: np.ndarray
    origin: float
    step: float


@njit(cache=True, nogil=True)
def _prepare_packed(offsets, mz, intensity, precursor, noise_cutoff,
                    remove_precursor, offset_to_precursor, merge_distance,
                    minimum_gap, merge, weighted):
    """Filter, enforce separation, and entropy-weight packed raw spectra.

    Close peaks are merged around fixed representatives selected by descending
    intensity (lower m/z breaks ties). Representatives do not move. Consequently
    surviving representatives are more than ``merge_distance`` apart.
    """
    out_mz = np.empty(mz.size, dtype=mz.dtype)
    out_int = np.empty(intensity.size, dtype=intensity.dtype)
    out_offsets = np.empty(offsets.size, dtype=np.int64)
    out_offsets[0] = 0
    write = 0
    filtered = 0
    merged = 0
    for row in range(offsets.size - 1):
        start = write
        cutoff = precursor[row] + offset_to_precursor
        use_cutoff = remove_precursor and np.isfinite(precursor[row])
        previous = -np.inf
        for p in range(offsets[row], offsets[row + 1]):
            x, value = np.float64(mz[p]), np.float64(intensity[p])
            if (not np.isfinite(x) or not np.isfinite(value)
                    or x < 0 or value <= 0 or (use_cutoff and x > cutoff)):
                filtered += 1
                continue
            if x < previous:
                return out_offsets, out_mz, out_int, filtered, merged, row, 1
            previous = x
            out_mz[write] = mz[p]
            out_int[write] = intensity[p]
            write += 1

        # Scale before merging to avoid overflow when summing large intensities.
        # A common scale factor does not change relative filtering or weighting.
        if write > start:
            maximum = np.max(out_int[start:write])
            stop = write
            write = start
            for p in range(start, stop):
                value = np.float64(out_int[p]) / np.float64(maximum)
                if value < noise_cutoff:
                    filtered += 1
                    continue
                out_mz[write] = out_mz[p]
                out_int[write] = value
                write += 1

        close = False
        for p in range(start + 1, write):
            if np.float64(out_mz[p]) - np.float64(out_mz[p - 1]) <= minimum_gap:
                close = True
                break
        if close and not merge:
            return out_offsets, out_mz, out_int, filtered, merged, row, 2
        if close:
            values = out_int[start:write].astype(np.float64)
            coordinates = out_mz[start:write].copy()
            order = np.argsort(-values, kind="mergesort")
            consumed = np.zeros(values.size, dtype=np.bool_)
            representatives = np.zeros(values.size, dtype=np.float64)
            for position in order:
                if consumed[position]:
                    continue
                consumed[position] = True
                mass = np.float64(coordinates[position])
                total = values[position]
                left = position - 1
                while left >= 0 and mass - np.float64(coordinates[left]) <= merge_distance:
                    if not consumed[left]:
                        total += values[left]
                        consumed[left] = True
                    left -= 1
                right = position + 1
                while right < values.size and np.float64(coordinates[right]) - mass <= merge_distance:
                    if not consumed[right]:
                        total += values[right]
                        consumed[right] = True
                    right += 1
                representatives[position] = total
            before = write - start
            write = start
            # Values are divided by the largest merged value before casting.
            largest = np.max(representatives)
            for p in range(values.size):
                if representatives[p] > 0:
                    out_mz[write] = coordinates[p]
                    out_int[write] = representatives[p] / largest
                    write += 1
            merged += before - (write - start)

        if write > start:
            total = 0.0
            for p in range(start, write):
                total += np.float64(out_int[p])
            entropy = 0.0
            for p in range(start, write):
                probability = np.float64(out_int[p]) / total
                if probability > 0:
                    entropy -= probability * np.log(probability)
            power = 0.25 + 0.25 * entropy if weighted and entropy < 3.0 else 1.0
            weight_sum = 0.0
            for p in range(start, write):
                weight_sum += (np.float64(out_int[p]) / total) ** power
            stop = write
            write = start
            for p in range(start, stop):
                # Drop values that underflow in the requested storage dtype.
                value = 0.5 * (np.float64(out_int[p]) / total) ** power / weight_sum
                out_int[write] = value
                if out_int[write] > 0:
                    out_mz[write] = out_mz[p]
                    write += 1
                else:
                    filtered += 1
        out_offsets[row + 1] = write
    return out_offsets, out_mz[:write], out_int[:write], filtered, merged, -1, 0


@njit(cache=True, nogil=True)
def _validate_separation(offsets, mz, intensities, minimum_gap):
    """Check stored, positive peaks once before a reference index is used."""
    for row in range(offsets.size - 1):
        start, stop = offsets[row], offsets[row + 1]
        total = 0.0
        for p in range(start, stop):
            if not np.isfinite(mz[p]) or not np.isfinite(intensities[p]) or intensities[p] <= 0:
                return row
            if p > start and np.float64(mz[p]) - np.float64(mz[p - 1]) <= minimum_gap:
                return row
            total += np.float64(intensities[p])
        if stop > start and abs(total - 0.5) > 1e-5:
            return row
    return -1


@njit(cache=True, inline="always")
def _bisect(mz, value, starts, origin, step, side_right):
    """Binary search with a bounded, verified directory bracket."""
    low, high = 0, mz.size
    if starts.size:
        position = (value - origin) / step
        if np.isfinite(position) and 1 <= position < starts.size - 2:
            cell = int(position)
            low, high = int(starts[cell - 1]), int(starts[cell + 2])
            if ((low > 0 and mz[low - 1] >= value)
                    or (high < mz.size and mz[high] <= value)):
                low, high = 0, mz.size
    while low < high:
        middle = low + (high - low) // 2
        if mz[middle] < value or (side_right and mz[middle] == value):
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, nogil=True)
def _query_windows(query_mz, library_mz, tolerance, starts, origin, step):
    """Find exact Da windows using float64 differences of stored coordinates.

    Endpoint trimming, rather than a branch for every candidate, reproduces
    ``abs(float64(query) - float64(reference)) <= tolerance`` at boundaries.
    """
    bounds = np.empty((query_mz.size, 2), dtype=np.int64)
    for p in range(query_mz.size):
        mass = np.float64(query_mz[p])
        low = _bisect(library_mz, np.nextafter(mass - tolerance, -np.inf), starts, origin, step, False)
        high = _bisect(library_mz, np.nextafter(mass + tolerance, np.inf), starts, origin, step, True)
        while low < high and mass - np.float64(library_mz[low]) > tolerance:
            low += 1
        while high > low and np.float64(library_mz[high - 1]) - mass > tolerance:
            high -= 1
        bounds[p, 0], bounds[p, 1] = low, high
    return bounds


@njit(cache=True, nogil=True)
def _accumulate_short(out, offsets, query_intensity, library_intensity, terms,
                      spectrum_ids, bounds, first_row, last_row):
    """Process short windows without Python calls or per-library matching state."""
    for row in range(first_row, last_row):
        for p in range(offsets[row], offsets[row + 1]):
            low, high = bounds[p]
            if high - low >= _VECTOR_MIN_HITS:
                continue
            value = query_intensity[p]
            query_term = value * np.log2(value)
            for posting in range(low, high):
                mixed = value + library_intensity[posting]
                contribution = mixed * np.log2(mixed) - terms[posting] - query_term
                out[row, spectrum_ids[posting]] += contribution


@njit(cache=True, nogil=True)
def _scatter_add(out, spectrum_ids, contributions):
    """Accumulate one vectorized chunk; IDs are unique within a query window."""
    for p in range(spectrum_ids.size):
        out[spectrum_ids[p]] += contributions[p]


def _score_block(out, prepared, view, bounds, first_row, last_row):
    """Score disjoint rows; long windows use bounded, per-worker scratch space."""
    _accumulate_short(out, prepared.spec_offsets, prepared.spec_int,
                      view.intensity, view.terms, view.spectrum_ids,
                      bounds, first_row, last_row)
    first_peak, last_peak = prepared.spec_offsets[[first_row, last_row]]
    window_sizes = bounds[first_peak:last_peak, 1] - bounds[first_peak:last_peak, 0]
    large_peaks = np.flatnonzero(window_sizes >= _VECTOR_MIN_HITS) + first_peak
    if not large_peaks.size:
        return
    capacity = min(_VECTOR_CHUNK_SIZE, int(window_sizes.max()))
    mixed = np.empty(capacity, dtype=view.intensity.dtype)
    contributions = np.empty_like(mixed)
    rows = np.searchsorted(prepared.spec_offsets[1:], large_peaks, side="right")
    for row, peak in zip(rows, large_peaks, strict=True):
        low, high = bounds[peak]
        value = prepared.spec_int[peak]
        query_term = value * np.log2(value)
        for start in range(low, high, capacity):
            stop = min(start + capacity, high)
            mixture = mixed[:stop - start]
            delta = contributions[:stop - start]
            np.add(view.intensity[start:stop], value, out=mixture)
            np.log2(mixture, out=delta)
            np.multiply(mixture, delta, out=delta)
            np.subtract(delta, view.terms[start:stop], out=delta)
            np.subtract(delta, query_term, out=delta)
            _scatter_add(out[row], view.spectrum_ids[start:stop], delta)


class EntropySearch(BaseSimilarity):
    """Fast fragment entropy similarity for well-separated spectra.

    Each nonempty spectrum is filtered, entropy-weighted, and normalized to
    weighted intensity 0.5. Product m/z values are indexed globally; scores are
    accumulated directly from matching indexed peaks. Peak separation guarantees
    one-to-one matches without allocating peak-use state for every reference.

    The mathematical entropy contribution is unchanged. Native float32 arithmetic
    is used with ``dtype=np.float32``; small roundoff differences relative to a
    float64 accumulator are expected. This is not an approximate-logarithm method.

    Parameters
    ----------
    tolerance
        Maximum absolute fragment m/z difference in Da, inclusive. Matching uses
        float64 differences of the stored coordinates.
    max_tolerance
        Maximum supported search tolerance. Defaults to ``tolerance``. Build an
        index with a larger value to reuse it with smaller search tolerances.
    peak_separation
        ``"merge"`` merges close peaks around fixed, most-intense representatives
        using a radius of 2.1 times ``max_tolerance``. ``"raise"`` requires input
        peaks to already be separated by more than 2 times ``max_tolerance``.
        The check includes a small float64 safety margin. Merging changes spectra
        and is not guaranteed to reproduce another package's centroiding method.
    noise_cutoff
        Remove peaks below this fraction of the maximum remaining intensity,
        after optional precursor removal and before merging. Zero or None disables
        the filter. All nonfinite/negative masses and nonpositive intensities are
        discarded.
    remove_precursor
        Remove peaks above ``precursor_mz + offset_to_precursor``. Missing or
        nonpositive precursor values skip this operation for that spectrum.
    offset_to_precursor
        Additive precursor cutoff in Da.
    intensity_weighting
        Apply low-entropy intensity weighting. False gives unweighted entropy
        similarity. Normalization to intensity sum 0.5 is always applied.
    dtype
        Storage, accumulation, and output dtype: float32 (default) or float64.
    index_step
        Mass-directory spacing in Da. This only accelerates exact lookups; it
        does not bin peaks or change tolerance. Zero disables the directory.
    matching_mode, use_ppm
        Only ``matching_mode="fragment"`` and ``use_ppm=False`` are supported.
        Unsupported modes raise instead of silently selecting another algorithm.

    Notes
    -----
    ``search(queries, library_index)`` returns query rows and library columns.
    The existing persistent ``FlashIndex`` is used with a distinct preparation
    configuration. Index construction and validation occur outside reused search.
    No existing matchms similarity or saved index format is replaced.

    For apples-to-apples kernel comparisons, give all implementations the same
    separated peak arrays; use ``peak_separation="raise"`` here and disable
    additional filtering. With default merging enabled, score equality to the
    general one-to-one Entropy implementation is not guaranteed.

    See Li et al. (2021), doi:10.1038/s41592-021-01331-z, and Li and Fiehn (2023),
    doi:10.1038/s41592-023-02012-9, for the entropy score and indexed search method.
    """

    is_commutative = True
    score_datatype = np.float32
    score_fields = ("score",)

    def __init__(
        self,
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        *,
        max_tolerance: float | None = None,
        peak_separation: str = "merge",
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        intensity_weighting: bool = True,
        dtype=np.float32,
        index_step: float = 0.01,
        matching_mode: str = "fragment",
        use_ppm: bool = False,
    ):
        if matching_mode != "fragment" or use_ppm:
            raise ValueError("EntropySearch supports fragment matching in Da only.")
        self.tolerance = float(tolerance)
        self.max_tolerance = self.tolerance if max_tolerance is None else float(max_tolerance)
        if (not math.isfinite(self.tolerance) or self.tolerance < 0
                or not math.isfinite(self.max_tolerance)
                or self.max_tolerance < self.tolerance):
            raise ValueError("Require finite 0 <= tolerance <= max_tolerance.")
        if peak_separation not in ("merge", "raise"):
            raise ValueError("peak_separation must be 'merge' or 'raise'.")
        self.peak_separation = peak_separation
        self.noise_cutoff = float(noise_cutoff or 0.0)
        if not math.isfinite(self.noise_cutoff) or not 0 <= self.noise_cutoff <= 1:
            raise ValueError("noise_cutoff must be finite and in [0, 1].")
        self.remove_precursor = bool(remove_precursor)
        self.offset_to_precursor = float(offset_to_precursor)
        if not math.isfinite(self.offset_to_precursor):
            raise ValueError("offset_to_precursor must be finite.")
        self.intensity_weighting = bool(intensity_weighting)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64.")
        self.score_datatype = self.dtype
        self.index_step = float(index_step)
        if not math.isfinite(self.index_step) or self.index_step < 0:
            raise ValueError("index_step must be finite and nonnegative.")
        self.matching_mode = matching_mode
        self.use_ppm = False

    def to_dict(self) -> dict:
        """Return constructor parameters, excluding prepared arrays and caches."""
        return {"__Similarity__": type(self).__name__, "tolerance": self.tolerance,
                "max_tolerance": self.max_tolerance, "peak_separation": self.peak_separation,
                "noise_cutoff": self.noise_cutoff, "remove_precursor": self.remove_precursor,
                "offset_to_precursor": self.offset_to_precursor,
                "intensity_weighting": self.intensity_weighting, "dtype": self.dtype.name,
                "index_step": self.index_step}

    def _config(self) -> dict:
        return {"weighing_type": "entropy_separated", "preparation": _PREPARATION_FORMAT,
                "max_tolerance": self.max_tolerance, "peak_separation": self.peak_separation,
                "noise_cutoff": self.noise_cutoff, "remove_precursor": self.remove_precursor,
                "offset_to_precursor": self.offset_to_precursor,
                "intensity_weighting": self.intensity_weighting, "dtype": self.dtype.str,
                "normalize_to_half": True}

    def _settings(self) -> tuple:
        return tuple(self._config().items())

    def _minimum_gap(self) -> float:
        # Base the margin on tolerance, not the largest mass in the batch.
        # An unrelated outlier (possibly removed by the precursor filter) must
        # not change how ordinary peaks are merged. Subtraction of nearby
        # nonnegative floating-point coordinates is exact by Sterbenz's lemma;
        # the margin covers rounding at tolerance-scale coordinates/endpoints.
        if self.max_tolerance == 0:
            return 0.0
        doubled = 2.0 * self.max_tolerance
        return doubled + 8.0 * np.finfo(float).eps * doubled

    @staticmethod
    def _precursors(values, n: int) -> np.ndarray:
        if values is None:
            return np.full(n, np.nan)
        try:
            precursor = np.asarray(values, dtype=np.float64).copy()
        except (ValueError, TypeError):
            import pandas as pd
            precursor = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        if precursor.shape != (n,):
            raise ValueError("precursor_mz must have one value per spectrum.")
        precursor[~np.isfinite(precursor) | (precursor <= 0)] = np.nan
        return precursor

    def _prepare_arrays(self, offsets, mz, intensities, precursor) -> _PreparedEntropySpectra:
        offsets = np.asarray(offsets)
        if (offsets.ndim != 1 or not offsets.size or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Peak offsets must be sorted nonnegative integers starting at zero.")
        mz = np.ascontiguousarray(mz, dtype=self.dtype)
        intensities = np.ascontiguousarray(intensities, dtype=self.dtype)
        if mz.ndim != 1 or intensities.shape != mz.shape or offsets[-1] != mz.size:
            raise ValueError("Peak offsets, masses, and intensities have incompatible shapes.")
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        precursor = self._precursors(precursor, offsets.size - 1)
        gap = self._minimum_gap()
        distance = max(2.1 * self.max_tolerance, gap)
        result = _prepare_packed(
            offsets, mz, intensities, precursor, self.noise_cutoff,
            self.remove_precursor, self.offset_to_precursor, distance, gap,
            self.peak_separation == "merge", self.intensity_weighting,
        )
        out_offsets, out_mz, out_int, filtered, merged, bad_row, code = result
        if code:
            problem = "unsorted peaks" if code == 1 else "peaks separated by <= 2 * max_tolerance"
            raise ValueError(f"Spectrum row {bad_row} has {problem}; use peak_separation='merge' for close peaks.")
        # Compact reduced arrays so discarded peaks do not retain a large buffer.
        if out_mz.size < 0.8 * mz.size:
            out_mz, out_int = out_mz.copy(), out_int.copy()
        return _PreparedEntropySpectra(
            _readonly(out_offsets), _readonly(out_mz), _readonly(out_int),
            _readonly(precursor), self._settings(), mz.size, int(filtered), int(merged),
        )

    def prepare_peak_arrays(self, peaks, precursor_mz=None) -> _PreparedEntropySpectra:
        """Prepare an iterable of raw ``(n_peaks, 2)`` arrays without Spectrum objects.

        Useful for cross-package comparisons: the same cleaned, normalized peak
        arrays can be passed to MSEntropy and here. Do not entropy-weight them
        beforehand unless ``intensity_weighting=False`` is explicitly intended.
        """
        arrays = [np.asarray(p, dtype=self.dtype) for p in peaks]
        if any(p.ndim != 2 or p.shape[1] != 2 for p in arrays):
            raise ValueError("Every peak array must have shape (n_peaks, 2), including empty arrays.")
        offsets = np.r_[0, np.cumsum([len(p) for p in arrays], dtype=np.int64)]
        combined = np.concatenate(arrays) if arrays else np.empty((0, 2), dtype=self.dtype)
        return self._prepare_arrays(offsets, combined[:, 0], combined[:, 1], precursor_mz)

    def prepare_queries(self, spectra) -> _PreparedEntropySpectra:
        """Prepare a SpectraCollection or iterable of Spectrum without modifying it.

        CSR collections are read in packed form, avoiding reconstruction of
        individual Spectrum objects. Other collection backends use get_row().
        """
        if hasattr(spectra, "fragments") and hasattr(spectra, "metadata"):
            fragments = spectra.fragments
            metadata = spectra.metadata
            precursor = metadata.get("precursor_mz", None)
            array = getattr(fragments, "array", None)
            if getattr(array, "format", None) == "csr" and hasattr(fragments, "bin_to_mz"):
                return self._prepare_arrays(
                    array.indptr, fragments.bin_to_mz(array.indices), array.data, precursor,
                )
            peaks = [np.column_stack(fragments.get_row(i)) for i in range(len(spectra))]
            return self.prepare_peak_arrays(peaks, precursor)
        spectra = list(spectra)
        return self.prepare_peak_arrays(
            [s.peaks.to_numpy for s in spectra],
            [s.get("precursor_mz") for s in spectra],
        )

    def _check_prepared(self, prepared) -> None:
        if not isinstance(prepared, _PreparedEntropySpectra) or prepared.settings != self._settings():
            raise ValueError("Use this EntropySearch configuration's prepare_queries()/prepare_peak_arrays() output.")

    def build_index(self, spectra) -> FlashIndex:
        """Prepare spectra and construct an index, including all search caches."""
        return self.build_index_prepared(self.prepare_queries(spectra))

    def build_index_prepared(self, prepared: _PreparedEntropySpectra) -> FlashIndex:
        """Index already prepared spectra without another cleanup/weighting pass."""
        self._check_prepared(prepared)
        if prepared.n_specs >= 2**32:  # hardly necessary, but who knows...
            raise ValueError("EntropySearch supports fewer than 2**32 reference spectra.")
        owners = np.repeat(np.arange(prepared.n_specs, dtype=np.uint32), np.diff(prepared.spec_offsets))
        order = np.argsort(prepared.spec_mz, kind="stable")
        index = FlashIndex(
            n_specs=prepared.n_specs, dtype=self.dtype,
            peaks_mz=_readonly(prepared.spec_mz[order]),
            peaks_int=_readonly(prepared.spec_int[order]),
            peaks_spec_idx=_readonly(owners[order]),
            spec_offsets=prepared.spec_offsets, spec_mz=prepared.spec_mz,
            spec_int=prepared.spec_int, precursor_mz=prepared.precursor_mz,
            config=self._config(),
            metadata={"input_peaks": prepared.n_input_peaks,
                      "filtered_peaks": prepared.n_filtered_peaks,
                      "merged_peaks": prepared.n_merged_peaks},
        )
        self.prime_index(index)
        return index

    def _check_index(self, index: FlashIndex) -> None:
        if not isinstance(index, FlashIndex):
            raise TypeError("Expected search(query_spectra, library_index: FlashIndex).")
        if index.dtype != self.dtype or any(index.config.get(k) != v for k, v in self._config().items()):
            raise ValueError("Index was not prepared with compatible EntropySearch settings; build a separate index.")
        if not 0 <= self.tolerance <= self.max_tolerance:
            raise ValueError("Search tolerance must be within the index's configured maximum.")

    def prime_index(self, index: FlashIndex) -> _SearchView:
        """Create read-only runtime caches once; no library scan on cache reuse.

        build_index() and load_index() call this before returning. Existing
        generic FlashIndex archives need compatible EntropySearch preparation.
        """
        self._check_index(index)
        key = (_CACHE_PREFIX, self.index_step)
        if key in index._runtime_cache:
            return index._runtime_cache[key]
        gap = self._minimum_gap()
        bad_row = _validate_separation(index.spec_offsets, index.spec_mz, index.spec_int, gap)
        if bad_row >= 0:
            raise ValueError(f"Index spectrum {bad_row} fails positive, normalized, separated-peak validation.")
        mz = _readonly(index.peaks_mz)
        intensity = _readonly(index.peaks_int, self.dtype)
        terms = _readonly(intensity * np.log2(intensity))
        owners = _readonly(index.peaks_spec_idx, np.uint32)
        starts = np.empty(0, dtype=np.int64)
        origin, step = 0.0, 1.0
        if mz.size > 1 and self.index_step > 0:
            origin, span = float(mz[0]), float(mz[-1]) - float(mz[0])
            if span > 0 and math.isfinite(span):
                step = max(self.index_step, span / (_MAX_DIRECTORY_ENTRIES - 2),
                           8 * np.spacing(max(abs(origin), abs(float(mz[-1])), 1.0)))
                count = min(_MAX_DIRECTORY_ENTRIES, math.ceil(span / step) + 2)
                edges = origin + np.arange(count, dtype=np.float64) * step
                starts = np.searchsorted(mz, edges).astype(np.int64)
        view = _SearchView(mz, intensity, terms, owners, _readonly(starts), origin, step)
        index._runtime_cache[key] = view
        return view

    def search(self, query_spectra, library_index: FlashIndex, *, score_fields=None,
               progress_bar=True, n_jobs=1) -> Scores:
        """Search raw queries; output shape is (n_queries, n_library_spectra)."""
        self._check_index(library_index)
        self._resolve_score_fields(score_fields)
        return self.search_prepared(
            self.prepare_queries(query_spectra), library_index,
            score_fields=score_fields, progress_bar=progress_bar, n_jobs=n_jobs,
        )

    def search_prepared(self, query_spectra: _PreparedEntropySpectra, library_index: FlashIndex,
                        *, score_fields=None, progress_bar=False, n_jobs=1) -> Scores:
        """Score prepared queries; preprocessing and index construction are excluded.

        n_jobs=1 is serial. Positive values specify worker threads; -1 uses the
        available CPUs, capped at the query count. Workers have disjoint output
        rows and private vector workspaces. There is no implicit GPU execution.
        """
        self._check_prepared(query_spectra)
        self._resolve_score_fields(score_fields)
        if isinstance(n_jobs, (bool, np.bool_)) \
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be a positive integer or -1.")
        view = self.prime_index(library_index)
        prepared = query_spectra
        scores = np.zeros((prepared.n_specs, library_index.n_specs), dtype=self.dtype)
        if not scores.size or not prepared.n_peaks or not view.mz.size:
            return Scores({"score": scores})
        # Compile the vector scatter signature during even a small warmup search.
        _scatter_add(scores[0], view.spectrum_ids[:0], np.empty(0, dtype=self.dtype))
        bounds = _query_windows(prepared.spec_mz, view.mz, self.tolerance,
                                view.starts, view.origin, view.step)
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        workers = min(prepared.n_specs, cpu_count if n_jobs == -1 else int(n_jobs))
        boundaries = np.linspace(0, prepared.n_specs, workers + 1, dtype=int)
        blocks = list(pairwise(boundaries))
        if workers == 1:
            _score_block(scores, prepared, view, bounds, 0, prepared.n_specs)
        else:
            # Precompile both signatures before starting threads (not timed away).
            # Normal benchmark warmup should execute a representative search.
            _accumulate_short(scores, prepared.spec_offsets, prepared.spec_int,
                              view.intensity, view.terms, view.spectrum_ids, bounds, 0, 0)
            _scatter_add(scores[0], view.spectrum_ids[:0], np.empty(0, dtype=self.dtype))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_score_block, scores, prepared, view, bounds, a, b) for a, b in blocks]
                for future in tqdm(futures, disable=not progress_bar, desc="EntropySearch query blocks"):
                    future.result()
        return Scores({"score": scores})

    def matrix(self, spectra_1, spectra_2=None, score_fields=None, progress_bar=True, n_jobs=1) -> Scores:
        """Compute a complete dense matrix, including preparation and index build.

        Self-comparison prepares the input only once. Both triangle halves and
        the diagonal are computed. Rows correspond to spectra_1, columns to
        spectra_2 (or spectra_1 when omitted).
        """
        self._resolve_score_fields(score_fields)
        queries = self.prepare_queries(spectra_1)
        references = queries if spectra_2 is None or spectra_2 is spectra_1 else self.prepare_queries(spectra_2)
        index = self.build_index_prepared(references)
        return self.search_prepared(queries, index, score_fields=score_fields,
                                    progress_bar=progress_bar, n_jobs=n_jobs)

    def pair(self, spectrum_1, spectrum_2) -> np.ndarray:
        """Return one scalar entropy similarity using the same preparation rules."""
        scores = self.matrix([spectrum_1], [spectrum_2], progress_bar=False, n_jobs=1)
        return np.asarray(scores.to_array()[0, 0], dtype=self.dtype)

    def save_index(self, index: FlashIndex, filename: str | Path, *, overwrite=True) -> None:
        """Use the existing FlashIndex archive format; runtime caches are omitted."""
        self._check_index(index)
        index.save(filename, overwrite=overwrite)

    def load_index(self, filename: str | Path) -> FlashIndex:
        """Load and eagerly validate/cache an EntropySearch-compatible FlashIndex."""
        index = FlashIndex.load(filename)
        self.prime_index(index)
        return index

    def index_statistics(self, index: FlashIndex) -> dict[str, Any]:
        """Return storage/workload metadata, not a process peak-memory estimate."""
        view = self.prime_index(index)
        return {"n_reference": index.n_specs, "n_peaks": view.mz.size,
                "dtype": self.dtype.name, "spectrum_id_dtype": view.spectrum_ids.dtype.name,
                "directory_entries": view.starts.size, "directory_bytes": view.starts.nbytes,
                "directory_step": view.step if view.starts.size else 0.0,
                "cached_entropy_bytes": view.terms.nbytes,
                "prepared_statistics": dict(index.metadata)}
