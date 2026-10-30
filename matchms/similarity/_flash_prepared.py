"""Validated, read-only packed spectra shared by indexed similarity kernels."""
from dataclasses import dataclass
import numpy as np


def readonly(array: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Return a contiguous read-only view without changing the source's flags.

    A copy is made only if required for contiguity or dtype conversion. Callers
    must not mutate a writable alias of a shared array while scoring uses it.
    """
    result = np.ascontiguousarray(array, dtype=dtype).view()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PreparedSpectra:
    """Packed output of a similarity's ``prepare_queries`` method.

    ``spec_offsets[i:i + 2]`` delimits spectrum i in ``spec_mz`` and ``spec_int``.
    Coordinates are sorted within each spectrum, and intensities have already
    been cleaned and weighted. ``precursor_mz`` and ``spec_l2`` contain one entry
    per spectrum; unknown precursors are NaN. Entropy uses zero norm placeholders.

    ``settings`` records the preprocessing configuration. Scorers check it at
    search time without repeatedly scanning the arrays. Instances should be
    obtained from the similarity API rather than constructed directly.
    """

    spec_offsets: np.ndarray
    spec_mz: np.ndarray
    spec_int: np.ndarray
    precursor_mz: np.ndarray
    spec_l2: np.ndarray
    dtype: np.dtype
    settings: tuple

    @property
    def n_specs(self) -> int:
        """Number of spectra, including empty rows."""
        return self.spec_offsets.size - 1

    @property
    def n_peaks(self) -> int:
        """Total number of packed peaks."""
        return self.spec_mz.size

    def slice(self, start: int, stop: int) -> "PreparedSpectra":
        """Select consecutive rows, sharing the read-only peak buffers."""
        if not 0 <= start <= stop <= self.n_specs:
            raise ValueError("Invalid prepared-spectrum slice.")
        peak_start, peak_stop = self.spec_offsets[start], self.spec_offsets[stop]
        return PreparedSpectra(
            spec_offsets=readonly(self.spec_offsets[start:stop + 1] - peak_start),
            spec_mz=self.spec_mz[peak_start:peak_stop],
            spec_int=self.spec_int[peak_start:peak_stop],
            precursor_mz=self.precursor_mz[start:stop],
            spec_l2=self.spec_l2[start:stop],
            dtype=self.dtype,
            settings=self.settings,
        )


def pack_native(native, dtype: np.dtype, settings: tuple) -> PreparedSpectra:
    """Validate native collection preprocessing output before compiled scoring.

    This is a linear validation step at preparation time, not at each query.
    Supplied L2 norms are retained and converted once to float64. Invalid or
    nonpositive precursor metadata are represented as NaN in the packed result.
    """
    dtype = np.dtype(dtype)
    offsets = np.asarray(native.spec_offsets)
    mz = np.asarray(native.spec_mz)
    intensities = np.asarray(native.spec_int)
    precursor = np.asarray(native.precursor_mz)
    if offsets.ndim != 1 or offsets.size == 0 or offsets.dtype.kind not in "iu":
        raise ValueError("Prepared offsets must be a nonempty integer vector.")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Prepared offsets must start at zero and be nondecreasing.")
    if mz.ndim != 1 or intensities.shape != mz.shape or offsets[-1] != mz.size:
        raise ValueError("Prepared offsets/peak shapes are inconsistent.")
    if precursor.shape != (offsets.size - 1,):
        raise ValueError("Prepared precursor count does not match spectrum count.")
    if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensities)):
        raise ValueError("Prepared peak coordinates and intensities must be finite.")
    if np.any(mz < 0) or np.any(intensities < 0):
        raise ValueError("Prepared peak coordinates and intensities must be nonnegative.")
    if mz.size > 1:
        backwards = mz[1:] < mz[:-1]
        boundaries = offsets[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < mz.size)]
        backwards[boundaries - 1] = False
        if np.any(backwards):
            raise ValueError("Prepared peaks must be m/z-sorted within each spectrum.")

    norms = getattr(native, "spec_l2", None)
    norms = (
        np.zeros(offsets.size - 1, dtype=np.float64)
        if norms is None else np.asarray(norms, dtype=np.float64)
    )
    if norms.shape != precursor.shape or not np.all(np.isfinite(norms)) or np.any(norms < 0):
        raise ValueError("Invalid prepared L2 norms.")
    precursor = np.asarray(precursor, dtype=np.float64)
    precursor = np.where(np.isfinite(precursor) & (precursor > 0), precursor, np.nan)
    return PreparedSpectra(
        spec_offsets=readonly(offsets, np.int64),
        spec_mz=readonly(mz, dtype),
        spec_int=readonly(intensities, dtype),
        precursor_mz=readonly(precursor),
        spec_l2=readonly(norms),
        dtype=dtype,
        settings=settings,
    )


def empty_prepared(dtype: np.dtype, settings: tuple) -> PreparedSpectra:
    """Construct a valid zero-row packed representation."""
    dtype = np.dtype(dtype)
    return PreparedSpectra(
        spec_offsets=readonly(np.array([0], dtype=np.int64)),
        spec_mz=readonly(np.empty(0, dtype=dtype)),
        spec_int=readonly(np.empty(0, dtype=dtype)),
        precursor_mz=readonly(np.empty(0, dtype=np.float64)),
        spec_l2=readonly(np.empty(0, dtype=np.float64)),
        dtype=dtype,
        settings=settings,
    )
