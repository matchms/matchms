from typing import List, Optional
import numpy as np


# ===================== library index =====================

class _LibraryIndex:
    """
    Compact container for the concatenated (sorted) library peaks and, optionally,
    the neutral-loss view. All arrays are read-only in workers.
    """
    __slots__ = (
        "n_specs",
        "peaks_mz", "peaks_int", "peaks_spec_idx",
        "nl_mz", "nl_int", "nl_spec_idx", "nl_product_idx",
        "spec_l2",
        "precursor_mz",
        "dtype",
    )
    def __init__(self, dtype: np.dtype):
        self.n_specs = 0
        self.peaks_mz = None
        self.peaks_int = None
        self.peaks_spec_idx = None
        self.nl_mz = None
        self.nl_int = None
        self.nl_spec_idx = None
        self.nl_product_idx = None
        self.spec_l2 = None
        self.precursor_mz = None
        self.dtype = dtype

def _build_library_index(processed_peaks_list: List[np.ndarray],
                         precursor_mz_list: List[Optional[float]],
                         compute_neutral_loss: bool = False,
                         compute_l2_norm: bool = False,
                         dtype: np.dtype = np.float32) -> _LibraryIndex:
    """
    Build a global, sorted index over all *query* spectra peaks.

    The index concatenates all query peaks into flat arrays, then sorts by m/z.
    This allows each reference spectrum (a row) to scan efficiently using
    binary searches into the shared arrays. If `build_neutral_loss` is True, we
    also construct a parallel set of arrays for neutral-loss peaks (precursor - mz).

    Arrays created (all aligned/sorted):
      - peaks_mz : float[dtype], all concatenated product (fragment) m/z
      - peaks_int : float[dtype], matching intensities
      - peaks_spec_idx : int32, which query spectrum each peak originated from
      - nl_mz, nl_int, nl_spec_idx, nl_product_idx (optional):
          neutral-loss m/z for peaks with known precursor,
          the intensities/spec_idx aligned to nl_mz,
          and `nl_product_idx` maps back into peaks_mz positions (for hybrid rules)

    Returns
    -------
    _LibraryIndex
        Compact structure used by the row workers to accumulate scores quickly.
    """
    idx = _LibraryIndex(dtype)
    idx.n_specs = len(processed_peaks_list)

    # precursor array
    prec = np.full(idx.n_specs, np.nan, dtype=dtype)
    for k, pmz in enumerate(precursor_mz_list):
        if pmz is not None:
            prec[k] = pmz
    idx.precursor_mz = prec

    if compute_l2_norm:
        spec_l2 = np.zeros(idx.n_specs, dtype=dtype)

    counts = [p.shape[0] for p in processed_peaks_list]
    N = int(np.sum(counts))
    if N == 0:
        idx.peaks_mz = np.zeros(0, dtype=dtype)
        idx.peaks_int = np.zeros(0, dtype=dtype)
        idx.peaks_spec_idx = np.zeros(0, dtype=np.int32)
        if compute_neutral_loss:
            idx.nl_mz = np.zeros(0, dtype=dtype)
            idx.nl_int = np.zeros(0, dtype=dtype)
            idx.nl_spec_idx = np.zeros(0, dtype=np.int32)
            idx.nl_product_idx = np.zeros(0, dtype=np.int64)
        if compute_l2_norm:
            idx.spec_l2 = spec_l2
        return idx

    mz_flat = np.empty(N, dtype=dtype)
    int_flat = np.empty(N, dtype=dtype)
    spec_flat = np.empty(N, dtype=np.int32)

    write = 0
    for s_i, p in enumerate(processed_peaks_list):
        n = p.shape[0]
        if n == 0:
            continue
        mz_flat[write:write+n] = p[:, 0]
        int_flat[write:write+n] = p[:, 1]
        spec_flat[write:write+n] = s_i
        if compute_l2_norm:
            spec_l2[s_i] = np.sqrt(np.sum((p[:, 1]).astype(np.float64)**2, dtype=np.float64)).astype(dtype)
        write += n

    order = np.argsort(mz_flat)
    idx.peaks_mz = mz_flat[order]
    idx.peaks_int = int_flat[order]
    idx.peaks_spec_idx = spec_flat[order]
    product_pos = np.empty(N, dtype=np.int64)
    product_pos[order] = np.arange(N, dtype=np.int64)

    if compute_neutral_loss:
        pmz_per_peak = idx.precursor_mz[spec_flat]  # float32, NaN if unknown
        have_pmz = ~np.isnan(pmz_per_peak)
        src_idx = np.nonzero(have_pmz)[0]
        if src_idx.size == 0:
            idx.nl_mz = np.zeros(0, dtype=dtype)
            idx.nl_int = np.zeros(0, dtype=dtype)
            idx.nl_spec_idx = np.zeros(0, dtype=np.int32)
            idx.nl_product_idx = np.zeros(0, dtype=np.int64)
        else:
            nl_mz = (pmz_per_peak[src_idx] - mz_flat[src_idx]).astype(dtype, copy=False)
            nl_int = int_flat[src_idx]
            nl_spec = spec_flat[src_idx]
            nl_prod = product_pos[src_idx]

            order_nl = np.argsort(nl_mz)
            idx.nl_mz = nl_mz[order_nl]
            idx.nl_int = nl_int[order_nl]
            idx.nl_spec_idx = nl_spec[order_nl]
            idx.nl_product_idx = nl_prod[order_nl]

    if compute_l2_norm:
        idx.spec_l2 = spec_l2

    return idx


# ===================== preprocessing =====================

def _entropy_weight(intensities: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Apply entropy-based weighting to intensities as described by Li & Fiehn (2023).
    """
    # TODO --> not really necessary: intensities = np.maximum(intensities, 0.0)
    total = float(intensities.sum(dtype=np.float64))  # sum in high precision for stability
    if total <= 0.0:
        return intensities
    p = intensities / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.zeros_like(p, dtype=dtype)
        mask = p > 0
        logp[mask] = np.log2(p[mask]).astype(dtype, copy=False)

    # entropy in bits
    S = float((-p * logp).sum(dtype=np.float64))
    w = 1.0 if S >= 3.0 else (0.25 + 0.25 * S)
    return np.power(intensities, w).astype(dtype, copy=False)


def _clean_and_weight(peaks: np.ndarray,
                      precursor_mz: Optional[float],
                      remove_precursor: bool,
                      precursor_window: float,
                      noise_cutoff: float,
                      normalize_to_half: bool,
                      merge_within_da: float,
                      score_type: str,
                      dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Apply the Flash preprocessing rules to a (mz, intensity) peak list.

    Steps:
      1) (Optional) Remove all peaks at/above (precursor_mz - precursor_window).
      2) (Optional) Remove noise: keep peaks with intensity >= noise_cutoff * max(intensity).
      3) (Optional) Entropy-weight intensities (Li & Fiehn): raise intensities by a power derived 
         from spectrum entropy.
      4) (Optional) Merge peaks within a small m/z window by intensity-weighted centroid.
      5) (Optional) normalize intensities to sum to 0.5 (recommended in the paper).

    Parameters
    ----------
    peaks : ndarray of shape (N, 2)
        Column 0 = m/z, column 1 = intensity.
    precursor_mz : float or None
        Precursor m/z for the spectrum (if known).
    remove_precursor : bool
        If True, remove the precursor and near-precursor peaks.
    precursor_window : float
        Size (Da) to cut below the precursor (remove m/z > precursor_mz - window).
    noise_cutoff : float
        Fraction of max intensity to keep (0.01 means keep peaks >= 1% of max).
    normalize_to_half : bool
        If True, scale intensities so their sum is 0.5.
    merge_within_da : float
        If > 0, merge peaks that are within this m/z distance.
    score_type : str
        One of "cosine" or "entropy". Fragment intensities will be weighted accordingly.
    dtype : np.dtype
        Float dtype for outputs. Default is np.float32.

    Returns
    -------
    ndarray
        Cleaned array with columns [mz, intensity], dtype=dtype, sorted by m/z.
        May be empty with shape (0, 2) if everything is filtered away.
    """
    if peaks.size == 0:
        return np.empty((0, 2), dtype=dtype)

    mz = _as_dtype(peaks[:, 0], dtype)
    intensities = _as_dtype(peaks[:, 1], dtype)

    # (optional) remove precursor and nearby peaks
    if remove_precursor and (precursor_mz is not None):
        mask = mz <= (precursor_mz - precursor_window)
        mz, intensities = mz[mask], intensities[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    # (optional) remove noise peaks below a fraction of max intensity
    if noise_cutoff and noise_cutoff > 0.0:
        thr = intensities.max() * noise_cutoff
        mask = intensities >= thr
        mz, intensities = mz[mask], intensities[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    # (optional) entropy-weight intensities (for flash entropy score)
    intensities = _entropy_weight(intensities, dtype)

    # (optional) merge nearby peaks by intensity-weighted centroid
    if merge_within_da and merge_within_da > 0.0 and mz.size > 1:
        peaks = _merge_within(np.column_stack((mz, intensities)), merge_within_da)
        mz, intensities = peaks[:, 0], peaks[:, 1]

    # (optional) normalize intensities to sum to 0.5
    if normalize_to_half:
        s = float(intensities.sum(dtype=np.float64))
        if s > 0.0:
            intensities = (intensities * (0.5 / s)).astype(dtype, copy=False)

    return np.column_stack((mz, intensities))


# ===================== smaller helper functions =====================

def _as_dtype(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return a.astype(dtype, copy=False) if a.dtype == dtype else a.astype(dtype, copy=True)


def _merge_within(
        peaks: np.ndarray,
        max_delta_da: float) -> np.ndarray:
    """
    Merges peaks within `max_delta_da` of each other by intensity-weighted averaging.
    
    Parameters
    ---------- 
    peaks : np.ndarray
        2D array of shape (n_peaks, 2) with m/z values in first column and intensities in second column.
    max_delta_da : float
        Maximum difference in m/z values to consider peaks as the same (in Dalton).
    """
    if peaks.shape[0] <= 1 or max_delta_da <= 0.0:
        return peaks
    mz = peaks[:, 0]
    intensities = peaks[:, 1]
    new_mz = []
    new_int = []
    current_mz = mz[0]
    current_int = intensities[0]
    for k in range(1, mz.shape[0]):
        if (mz[k] - current_mz) <= max_delta_da:
            total = current_int + intensities[k]
            if total > 0:
                current_mz = (current_mz * current_int + mz[k] * intensities[k]) / total
                current_int = total
            else:
                current_mz = mz[k]
                current_int = 0.0
        else:
            new_mz.append(current_mz)
            new_int.append(current_int)
            current_mz = mz[k]
            current_int = intensities[k]
    new_mz.append(current_mz)
    new_int.append(current_int)
    out = np.column_stack((np.array(new_mz, dtype=mz.dtype),
                           np.array(new_int, dtype=intensities.dtype)))
    return out
