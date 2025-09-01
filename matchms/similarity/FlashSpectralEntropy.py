import multiprocessing as mp
import platform
from typing import List, Optional, Tuple
import numpy as np
from sparsestack import StackedSparseArray
from tqdm import tqdm
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class FlashSpectralEntropy(BaseSimilarity):
    """
    Flash entropy similarity (Li & Fiehn, 2023) with a fast .matrix() that
    builds a library-wide index over 'queries' and streams all 'references'
    through it.

    Key options:
      - mode: 'fragment', 'neutral_loss', or 'hybrid' (fragment-priority).
      - tolerance in Da or symmetric ppm (use_ppm=True).
      - cleanup: remove precursor & > (precursor_mz - 1.6), 1% noise removal,
                 entropy weighting, normalize ∑I' = 0.5, optional within-peak merge.

    Notes:
      - .pair() works but is not the fast path. Use .matrix().
      - For identity-search behavior, pass identity_precursor_tolerance (Da or ppm).
    
    Parameters
    ----------
    tolerance:
        Matching tolerance in Da or ppm (use_ppm=True). Default is 0.02.
    use_ppm:
        If True, interpret `tolerance` as parts-per-million. Default is False.
    mode:
        Matching mode: 'fragment', 'neutral_loss', or 'hybrid' (default is 'fragment').
    remove_precursor:
        If True, remove precursor peak and peaks within precursor_window.
        Default is True.
    precursor_window:
        If remove_precursor is True, remove peaks within this window around the precursor
        m/z. Default is 1.6 Da (as suggested by Li & Fiehn(2023)).
    noise_cutoff:
        If > 0, remove peaks with intensities below this fraction of the maximum intensity.
        Default is 0.01 (1%).
    normalize_to_half:
        If True, normalize intensities such that the sum of intensities is 0.5.
        Default is True.
    merge_within:
        If > 0, merge peaks within this distance (in Da) to a single peak.
        Default is 0.05 Da.
    identity_precursor_tolerance:
        If not None, enforce identity search behavior by requiring the precursor m/z
        of the query to be within this tolerance of the reference precursor m/z.
    identity_use_ppm:
        If True, interpret `identity_precursor_tolerance` as ppm. Default is False.
    dtype:
        Data type for the output scores. Default is np.float32.

    """
    is_commutative = True
    score_datatype = np.float32

    def __init__(self,
                 tolerance: float = 0.02,
                 use_ppm: bool = False,
                 mode: str = "fragment",               # 'fragment' | 'neutral_loss' | 'hybrid'
                 remove_precursor: bool = True,
                 precursor_window: float = 1.6,
                 noise_cutoff: float = 0.01,
                 normalize_to_half: bool = True,
                 merge_within: float = 0.05,
                 identity_precursor_tolerance: Optional[float] = None,
                 identity_use_ppm: bool = False,
                 dtype: np.dtype = np.float32):
        if mode not in ("fragment", "neutral_loss", "hybrid"):
            raise ValueError("mode must be 'fragment', 'neutral_loss', or 'hybrid'")
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.mode = mode
        self.remove_precursor = remove_precursor
        self.precursor_window = precursor_window
        self.noise_cutoff = noise_cutoff
        self.normalize_to_half = normalize_to_half
        self.merge_within = merge_within
        self.identity_precursor_tolerance = identity_precursor_tolerance
        self.identity_use_ppm = identity_use_ppm
        self.dtype = dtype
        # sync default BaseSimilarity
        self.score_datatype = dtype

    # ---- per-pair (not parallel path) ----
    def _prepare(self, spectrum: SpectrumType) -> Tuple[np.ndarray, Optional[float]]:
        arr = spectrum.peaks.to_numpy
        pmz = spectrum.metadata.get("precursor_mz", None)
        cleaned = _clean_and_weight(arr, pmz,
                                    remove_precursor=self.remove_precursor,
                                    precursor_window=self.precursor_window,
                                    noise_cutoff=self.noise_cutoff,
                                    normalize_to_half=self.normalize_to_half,
                                    merge_within_da=self.merge_within,
                                    dtype=self.dtype)
        return cleaned, (None if pmz is None else float(pmz))

    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
        """
        Compute Flash entropy similarity for a single (reference, query) pair.
        Uses the same preprocessing and scoring logic as the matrix path, but
        builds a tiny 1-spectrum library from the query.
        """
        # preprocess both spectra
        A, pmzA = self._prepare(reference)
        B, pmzB = self._prepare(query)
        if A.size == 0 or B.size == 0:
            return np.asarray(0.0, dtype=self.dtype)
    
        # build 1-spec library index from the query (B)
        lib = _LibraryIndex(self.dtype)
        lib.n_specs = 1
        lib.peaks_mz = B[:, 0]
        lib.peaks_int = B[:, 1]
        lib.peaks_spec_idx = np.zeros(B.shape[0], dtype=np.int32)
    
        if self.mode in ("neutral_loss", "hybrid") and (pmzB is not None):
            nl_mz = (float(pmzB) - B[:, 0]).astype(self.dtype, copy=False)
            order = np.argsort(nl_mz)
            lib.nl_mz = nl_mz[order]
            lib.nl_int = B[:, 1][order]
            lib.nl_spec_idx = np.zeros(B.shape[0], dtype=np.int32)
            lib.nl_product_idx = order.astype(np.int64, copy=False)
        else:
            lib.nl_mz = np.zeros(0, dtype=self.dtype)
            lib.nl_int = np.zeros(0, dtype=self.dtype)
            lib.nl_spec_idx = np.zeros(0, dtype=np.int32)
            lib.nl_product_idx = np.zeros(0, dtype=np.int64)
    
        lib.precursor_mz = np.array(
            [float(pmzB) if (pmzB is not None) else np.nan],
            dtype=self.dtype
        )
    
        # compute scores into a 1-length buffer (single library spectrum)
        scores = np.zeros(1, dtype=self.dtype)
    
        # fragment path
        _accumulate_fragment_row_numba(
            scores,
            A[:, 0].astype(self.dtype, copy=False),
            A[:, 1].astype(self.dtype, copy=False),
            lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
            float(self.tolerance), bool(self.use_ppm)
        )
    
        # neutral-loss / hybrid path (if enabled & query has precursor)
        if self.mode in ("neutral_loss", "hybrid") and (pmzA is not None):
            prefer_frag = (self.mode == "hybrid")
    
            if prefer_frag:
                # precompute product-peak windows for ALL reference peaks
                prod_min = np.empty(A.shape[0], dtype=np.int64)
                prod_max = np.empty(A.shape[0], dtype=np.int64)
                for k in range(A.shape[0]):
                    mz1 = float(A[k, 0])
                    hw = _search_window_halfwidth_nb(mz1, float(self.tolerance), bool(self.use_ppm))
                    lo = mz1 - hw
                    hi = mz1 + hw
                    prod_min[k] = np.searchsorted(lib.peaks_mz, lo, side='left')
                    prod_max[k] = np.searchsorted(lib.peaks_mz, hi, side='right')
            else:
                prod_min = np.empty(0, dtype=np.int64)
                prod_max = np.empty(0, dtype=np.int64)
    
            q_pmz_val = float(pmzA) if (pmzA is not None) else np.nan
    
            _accumulate_nl_row_numba(
                scores,
                A[:, 0].astype(self.dtype, copy=False),
                A[:, 1].astype(self.dtype, copy=False),
                q_pmz_val,
                lib.nl_mz, lib.nl_int, lib.nl_spec_idx, lib.nl_product_idx,
                lib.peaks_mz, lib.peaks_spec_idx,
                float(self.tolerance), bool(self.use_ppm),
                prefer_frag,
                prod_min, prod_max
            )
    
        # identity gate (optional)
        if (self.identity_precursor_tolerance is not None) and (pmzA is not None):
            tol = float(self.identity_precursor_tolerance)
            lib_pmz = float(lib.precursor_mz[0])
            if self.identity_use_ppm:
                allow = abs(lib_pmz - pmzA) <= (tol * 1e-6 * 0.5 * (lib_pmz + pmzA))
            else:
                allow = abs(lib_pmz - pmzA) <= tol
            if not (allow and np.isfinite(lib_pmz)):
                scores[0] = self.dtype.type(0.0)
    
        return np.asarray(scores[0], dtype=self.dtype)

    # ---- FAST + PARALLEL ----
    def matrix(self,
               references: List[SpectrumType],
               queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False,
               n_jobs: int = -1) -> np.ndarray:
        """
        Calculate matrix of Flash entropy similarity scores.
        
        Parameters:
        ----------
        references:
            List of reference spectra.
        queries:
            List of query spectra.
        array_type:
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a SparseStacked COO-style array.
        is_symmetric:
            If True, the matrix will be symmetric (i.e., references and queries must have the same length).
            Here has no consequence on runtime.
        n_jobs:
            Number of parallel jobs to run.
            Default is set to -1, which means that all available CPUs minus one will be used.
        """
        n_rows = len(references)
        n_cols = len(queries)

        if array_type not in ("numpy", "sparse"):
            raise ValueError("array_type must be 'numpy' or 'sparse'.")
        if is_symmetric and n_rows != n_cols:
            raise ValueError("is_symmetric=True requires same #rows and #cols.")

        # ---- Windows safety fallback ----
        if platform.system() == "Windows" and n_jobs not in (None, 1, 0):
            print("FlashSpectralEntropy.matrix: n_jobs != 1 is not yet implemented on Windows; "
                  "falling back to n_jobs=1.")
            n_jobs = 1

        # 1) preprocess LIBRARY once
        lib_proc = []
        lib_pmz = []
        for s in queries:
            arr = s.peaks.to_numpy
            pmz = s.metadata.get("precursor_mz", None)
            cleaned = _clean_and_weight(arr, pmz,
                                        remove_precursor=self.remove_precursor,
                                        precursor_window=self.precursor_window,
                                        noise_cutoff=self.noise_cutoff,
                                        normalize_to_half=self.normalize_to_half,
                                        merge_within_da=self.merge_within,
                                        dtype=self.dtype)
            lib_proc.append(cleaned)
            lib_pmz.append(None if pmz is None else float(pmz))

        build_nl = (self.mode in ("neutral_loss", "hybrid"))
        lib = _build_library_index(lib_proc, lib_pmz, build_neutral_loss=build_nl, dtype=self.dtype)

        # 2) prepare output
        out = np.zeros((n_rows, n_cols), dtype=self.dtype)

        # 3) prepare row inputs (queries) – we preprocess each reference spectrum here
        row_inputs = []
        for i, ref in enumerate(references):
            A, pmzA = self._prepare(ref)
            row_inputs.append((i, A, pmzA))

        # 4) configuration passed to workers
        cfg = dict(
            tol=float(self.tolerance),
            use_ppm=bool(self.use_ppm),
            mode=self.mode,
            build_nl=build_nl,
            iden_tol=(None if self.identity_precursor_tolerance is None else float(self.identity_precursor_tolerance)),
            iden_use_ppm=bool(self.identity_use_ppm),
        )

        # 5) run — sequential or parallel
        if n_jobs in (None, 1, 0):
            # sequential
            _set_globals(lib, cfg)
            iterator = row_inputs
            worker = _row_task_dense #  if array_type == "numpy" else _row_task_sparse (TODO?)
            for item in tqdm(iterator, total=n_rows, desc="Flash entropy (matrix)"):
                row_idx, row = worker(item)
                # optional sanity check while debugging:
                # assert 0 <= row_idx < n_rows, (row_idx, n_rows)
                out[row_idx, :] = row
                
            if array_type == "numpy":
                return out
            elif array_type == "sparse":
                scores_array = StackedSparseArray(n_rows, n_cols)
                scores_array.add_dense_matrix(out.astype(self.score_datatype), "")
                return scores_array
            raise NotImplementedError("Output array type is unknown or not yet implemented.")

        # Attempt Unix fork-based parallelism (not available on Windows; we already guarded above)
        n_cpus = mp.cpu_count()
        if n_jobs < 0:
            n_jobs = max(1, n_cpus + 1 + n_jobs)  # e.g., -1 -> n_cpus-1
        n_jobs = max(1, min(n_jobs, n_cpus))

        start_methods = mp.get_all_start_methods()
        use_fork = ("fork" in start_methods) and (platform.system() != "Windows")

        if use_fork:
            ctx = mp.get_context("fork")
            _set_globals(lib, cfg)  # make available pre-fork
            worker = _row_task_dense  # (TODO: sparse path)
            with ctx.Pool(processes=n_jobs) as pool:
                for result in tqdm(pool.imap(worker, row_inputs, chunksize=8),
                                   total=n_rows, desc=f"Flash entropy (parallel x{n_jobs})"):
                    i, row = result
                    out[i, :] = row
            if array_type == "numpy":
                return out
            elif array_type == "sparse":
                scores_array = StackedSparseArray(n_rows, n_cols)
                scores_array.add_dense_matrix(out.astype(self.score_datatype), "")
                return scores_array
            raise NotImplementedError("Output array type is unknown or not yet implemented.")

        # If fork is not available (e.g., certain environments), fall back to sequential with a note.
        print("FlashSpectralEntropy.matrix: parallel execution requires 'fork'; "
              "falling back to n_jobs=1.")
        _set_globals(lib, cfg)
        for item in tqdm(row_inputs, total=n_rows, desc="Flash entropy (matrix)"):
            i, row = _row_task_dense(item)
            out[i, :] = row
        if array_type == "numpy":
            return out
        elif array_type == "sparse":
            scores_array = StackedSparseArray(n_rows, n_cols)
            scores_array.add_dense_matrix(out.astype(self.score_datatype), "")
            return scores_array
        raise NotImplementedError("Output array type is unknown or not yet implemented.")


# ===================== helpers =====================

def _as_dtype(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return a.astype(dtype, copy=False) if a.dtype == dtype else a.astype(dtype, copy=True)

def _xlog2_vec(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    # stable x*log2(x) with x=0 -> 0, in requested dtype (float32 default)
    out = np.zeros_like(x, dtype=dtype)
    mask = x > 0
    if np.any(mask):
        out[mask] = (x[mask] * np.log2(x[mask])).astype(dtype, copy=False)
    return out

def _xlog2_scalar(x: float, dtype: np.dtype) -> float:
    if x <= 0.0:
        return 0.0
    return x * np.log2(x)

def _within_tol(m1: float, m2: float, tol: float, use_ppm: bool, dtype: np.dtype) -> bool:
    if not use_ppm:
        return abs(m1 - m2) <= tol
    return abs(m1 - m2) <= tol * 1e-6 * (0.5 * (m1 + m2))

def _search_window_halfwidth(m: float, tol: float, use_ppm: bool, dtype: np.dtype) -> float:
    if not use_ppm:
        return tol
    c = tol * 1e-6
    denom = 1.0 - 0.5 * c
    return (c * m) / denom if denom > 0 else (c * m * 2.0)


# ===================== preprocessing (Flash rules) =====================

def _entropy_weight(intensities: np.ndarray, dtype: np.dtype) -> np.ndarray:
    intensities = _as_dtype(np.maximum(intensities, 0.0), dtype)
    total = float(intensities.sum(dtype=np.float64))  # sum in high precision for stability
    if total <= 0.0:
        return intensities
    p = _as_dtype(intensities / total, dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.zeros_like(p, dtype=dtype)
        mask = p > 0
        logp[mask] = np.log2(p[mask]).astype(dtype, copy=False)
    # entropy in bits
    S = float((-p * logp).sum(dtype=np.float64))
    w = 1.0 if S >= 3.0 else (0.25 + 0.25 * S)
    return np.power(intensities, w).astype(dtype, copy=False)

def _merge_within(peaks: np.ndarray, max_delta_da: float, dtype: np.dtype) -> np.ndarray:
    if peaks.shape[0] <= 1 or max_delta_da <= 0.0:
        return _as_dtype(peaks, dtype)
    mz = peaks[:, 0].astype(dtype, copy=True)
    inten = peaks[:, 1].astype(dtype, copy=True)
    new_mz = []
    new_int = []
    current_mz = mz[0]
    current_int = inten[0]
    for k in range(1, mz.shape[0]):
        if (mz[k] - current_mz) <= max_delta_da:
            total = current_int + inten[k]
            if total > 0:
                current_mz = (current_mz * current_int + mz[k] * inten[k]) / total
                current_int = total
            else:
                current_mz = mz[k]
                current_int = 0.0
        else:
            new_mz.append(float(current_mz))
            new_int.append(float(current_int))
            current_mz = mz[k]
            current_int = inten[k]
    new_mz.append(float(current_mz))
    new_int.append(float(current_int))
    out = np.column_stack((np.array(new_mz, dtype=dtype), np.array(new_int, dtype=dtype)))
    return out

def _clean_and_weight(peaks: np.ndarray,
                      precursor_mz: Optional[float],
                      remove_precursor: bool,
                      precursor_window: float,
                      noise_cutoff: float,
                      normalize_to_half: bool,
                      merge_within_da: float,
                      dtype: np.dtype) -> np.ndarray:
    """
    Apply the Flash preprocessing rules to a (mz, intensity) peak list.

    Steps:
      1) (Optional) Remove all peaks at/above (precursor_mz - precursor_window).
      2) (Optional) Remove noise: keep peaks with intensity >= noise_cutoff * max(intensity).
      3) Entropy-weight intensities (Li & Fiehn): raise intensities by a power derived from spectrum entropy.
      4) (Optional) Merge peaks within a small m/z window by intensity-weighted centroid.
      5) Sort by m/z and (optional) normalize intensities to sum to 0.5 (recommended in the paper).

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
    dtype : np.dtype
        Float dtype for outputs, usually np.float32.

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

    if remove_precursor and (precursor_mz is not None):
        mask = mz <= (precursor_mz - precursor_window)
        mz, inten = mz[mask], intensities[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    if noise_cutoff and noise_cutoff > 0.0:
        #thr = dtype.type(mz.dtype.type(inten.max()) * noise_cutoff)
        thr = intensities.max() * noise_cutoff
        mask = intensities >= thr
        mz, intensities = mz[mask], intensities[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    intensities = _entropy_weight(intensities, dtype)

    if merge_within_da and merge_within_da > 0.0 and mz.size > 1:
        peaks = _merge_within(np.column_stack((mz, intensities)), merge_within_da, dtype)
        mz, intensities = peaks[:, 0], peaks[:, 1]
    else:
        order = np.argsort(mz)
        mz, inten = mz[order], intensities[order]

    s = float(intensities.sum(dtype=np.float64))
    if s > 0.0 and normalize_to_half:
        intensities = (intensities * (0.5 / s)).astype(dtype, copy=False)

    return np.column_stack((mz, intensities))


# ===================== library index =====================

class _LibraryIndex:
    __slots__ = (
        "n_specs",
        "peaks_mz", "peaks_int", "peaks_spec_idx",
        "nl_mz", "nl_int", "nl_spec_idx", "nl_product_idx",
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
        self.precursor_mz = None
        self.dtype = dtype

def _build_library_index(processed_peaks_list: List[np.ndarray],
                         precursor_mz_list: List[Optional[float]],
                         build_neutral_loss: bool,
                         dtype: np.dtype) -> _LibraryIndex:
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
    # store precursor m/z as float32 (NaN when unknown)
    prec = np.full(idx.n_specs, np.nan, dtype=dtype)
    for k, pmz in enumerate(precursor_mz_list):
        if pmz is not None:
            prec[k] = pmz
    idx.precursor_mz = prec

    counts = [p.shape[0] for p in processed_peaks_list]
    N = int(np.sum(counts))
    if N == 0:
        idx.peaks_mz = np.zeros(0, dtype=dtype)
        idx.peaks_int = np.zeros(0, dtype=dtype)
        idx.peaks_spec_idx = np.zeros(0, dtype=np.int32)
        if build_neutral_loss:
            idx.nl_mz = np.zeros(0, dtype=dtype)
            idx.nl_int = np.zeros(0, dtype=dtype)
            idx.nl_spec_idx = np.zeros(0, dtype=np.int32)
            idx.nl_product_idx = np.zeros(0, dtype=np.int64)
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
        write += n

    order = np.argsort(mz_flat)
    idx.peaks_mz = mz_flat[order]
    idx.peaks_int = int_flat[order]
    idx.peaks_spec_idx = spec_flat[order]
    product_pos = np.empty(N, dtype=np.int64)
    product_pos[order] = np.arange(N, dtype=np.int64)

    if build_neutral_loss:
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

    return idx


# --- Numba-accelerated accumulators ------------------------------------------
from numba import njit


@njit(cache=True, nogil=True)
def _search_window_halfwidth_nb(m: float, tol: float, use_ppm: bool) -> float:
    """
    Compute half-width of a symmetric search window around a value `mass`.

    If `use_ppm` is False: returns `tol` (Da half-width).
    If `use_ppm` is True : uses symmetric-ppm definition:
        |m2 - m1| <= tol[ppm] * 1e-6 * 0.5 * (m1 + m2)
    which corresponds to a half-window of approximately (tol * 1e-6 * mass),
    corrected for symmetry.
    """
    if not use_ppm:
        return tol
    c = tol * 1e-6
    denom = 1.0 - 0.5 * c
    return (c * m) / denom if denom > 0.0 else (c * m * 2.0)

@njit(cache=True, nogil=True)
def _xlog2_scalar_nb(x: float) -> float:
    """
    Numerically stable x * log2(x) for scalar x with x <= 0 mapped to 0.
    """
    if x <= 0.0:
        return 0.0
    return x * np.log2(x)

@njit(cache=True, nogil=True)
def _accumulate_fragment_row_numba(
    scores: np.ndarray,
    query_mz: np.ndarray, query_intensity: np.ndarray,
    lib_mz: np.ndarray, lib_intensity: np.ndarray, lib_spec_index: np.ndarray,
    tol: float, use_ppm: bool
) -> None:
    """
    Accumulate fragment-based contributions for a single row (one reference spectrum).

    For each query peak (m/z, Iq), find all library peaks whose m/z lies within
    the symmetric-ppm/Da tolerance window around the query m/z. For each match,
    add the incremental entropy term to `scores[col]`, where `col` is the index
    of the library spectrum that the matched library peak belongs to.

    Parameters
    ----------
    scores : float[dtype], shape (n_library_spectra,)
        Output buffer, accumulated in-place.
    query_mz, query_intensity : float[dtype]
        Peaks of the reference spectrum being compared (this row).
    lib_mz, lib_intensity : float[dtype]
        Global concatenated *query* (library) product peaks, sorted by m/z.
    lib_spec_index : int32
        For each entry of lib_mz, the source spectrum index (column) in the library.
    tol : float
        Mass tolerance (Da or ppm).
    use_ppm : bool
        Whether to treat `tol` as ppm.
    """
    n_q = query_mz.shape[0]
    for q_idx in range(n_q):
        mz_q = float(query_mz[q_idx])
        Iq = float(query_intensity[q_idx])
        if Iq <= 0.0:
            continue

        half_width = _search_window_halfwidth_nb(mz_q, tol, use_ppm)
        lo = mz_q - half_width
        hi = mz_q + half_width

        a = np.searchsorted(lib_mz, lo, side='left')
        b = np.searchsorted(lib_mz, hi, side='right')
        if a >= b:
            continue

        for j in range(a, b):
            mz_lib = float(lib_mz[j])
            if use_ppm:
                if abs(mz_lib - mz_q) > (tol * 1e-6) * 0.5 * (mz_lib + mz_q):
                    continue
            else:
                if abs(mz_lib - mz_q) > tol:
                    continue

            Ilib = float(lib_intensity[j])
            col = int(lib_spec_index[j])
            incr = _xlog2_scalar_nb(Ilib + Iq) - _xlog2_scalar_nb(Iq) - _xlog2_scalar_nb(Ilib)
            scores[col] += incr


@njit(cache=True, nogil=True)
def _in_any_fragment_window(prod_idx: int, prod_min: np.ndarray, prod_max: np.ndarray) -> bool:
    # true if prod_idx falls into ANY [prod_min[k], prod_max[k]) interval
    # (arrays come precomputed; when prefer_fragments=False, caller passes size 0 arrays)
    for k in range(prod_min.shape[0]):
        if prod_idx >= prod_min[k] and prod_idx < prod_max[k]:
            return True
    return False

@njit(cache=True, nogil=True)
def _spec_in_fragment_window(cols_target: int,
                             peaks_spec: np.ndarray, ap: int, bp: int) -> bool:
    # true if cols_target appears in peaks_spec[ap:bp]
    for t in range(ap, bp):
        if int(peaks_spec[t]) == cols_target:
            return True
    return False


@njit(cache=True, nogil=True)
def _accumulate_nl_row_numba(scores: np.ndarray,
                             q_mz: np.ndarray, q_int: np.ndarray, q_pmz_val: float,  # pass np.nan if unknown
                             nl_mz: np.ndarray, nl_int: np.ndarray, nl_spec: np.ndarray, nl_prod_idx: np.ndarray,
                             peaks_mz: np.ndarray, peaks_spec: np.ndarray,
                             tol: float, use_ppm: bool,
                             prefer_fragments: bool,
                             prod_min: np.ndarray, prod_max: np.ndarray) -> None:
    """
    Numba version of _accumulate_nl_row.

    Notes:
      - Pass q_pmz_val=np.nan to indicate 'no precursor' (early return).
      - When prefer_fragments is True, prod_min/prod_max must be precomputed per query peak
        (length == len(q_mz)); otherwise pass empty (len 0) arrays.
    """
    if not (nl_mz.size > 0 and q_mz.size > 0):
        return
    if np.isnan(q_pmz_val):
        return

    for i in range(q_mz.shape[0]):
        Iq = float(q_int[i])
        if Iq <= 0.0:
            continue

        loss = float(q_pmz_val) - float(q_mz[i])
        mz_tolerance = _search_window_halfwidth_nb(loss, tol, use_ppm)
        lo_mz = loss - mz_tolerance
        hi_mz = loss + mz_tolerance

        a = np.searchsorted(nl_mz, lo_mz, side='left')
        b = np.searchsorted(nl_mz, hi_mz, side='right')
        if a >= b:
            continue

        # For RULE 2 we need the fragment window for THIS peak:
        ap = 0
        bp = 0
        if prefer_fragments:
            mz1 = float(q_mz[i])
            frag_hw = _search_window_halfwidth_nb(mz1, tol, use_ppm)
            flo = mz1 - frag_hw
            fhi = mz1 + frag_hw
            ap = np.searchsorted(peaks_mz, flo, side='left')
            bp = np.searchsorted(peaks_mz, fhi, side='right')

        for j in range(a, b):
            nl2 = float(nl_mz[j])
            # symmetric ppm / Da check
            if use_ppm:
                if abs(nl2 - loss) > (tol * 1e-6) * 0.5 * (nl2 + loss):
                    continue
            else:
                if abs(nl2 - loss) > tol:
                    continue

            lib = float(nl_int[j])
            col = int(nl_spec[j])

            # Hybrid priority rules
            if prefer_fragments:
                # RULE 1: drop if product peak falls in ANY fragment window of the query spectrum
                if _in_any_fragment_window(int(nl_prod_idx[j]), prod_min, prod_max):
                    continue

                # RULE 2 (per-peak): if this query peak also fragment-matches the same library spectrum, drop
                if ap < bp and _spec_in_fragment_window(col, peaks_spec, ap, bp):
                    continue

            v = _xlog2_scalar_nb(lib + Iq) - _xlog2_scalar_nb(Iq) - _xlog2_scalar_nb(lib)
            scores[col] += v



# ===================== worker plumbing =====================

# Globals visible to workers
_G_LIB = None
_G_CFG = None

def _set_globals(lib_obj, cfg):
    global _G_LIB, _G_CFG
    _G_LIB = lib_obj
    _G_CFG = cfg

def _row_task_dense(args):
    """Compute one row; return (row_index, dense_row_float32)."""
    row_idx, q_arr, q_pmz = args
    cfg = _G_CFG
    lib = _G_LIB
    dtype = lib.dtype
    scores = np.zeros(lib.n_specs, dtype=dtype)

    _accumulate_fragment_row_numba(scores,
                                   q_arr[:, 0], q_arr[:, 1],
                                   lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
                                   float(cfg["tol"]), bool(cfg["use_ppm"]))

    prefer_frag = (cfg["mode"] == "hybrid")
    if prefer_frag:
        prod_min = np.empty(q_arr.shape[0], dtype=np.int64)
        prod_max = np.empty(q_arr.shape[0], dtype=np.int64)
        for k in range(q_arr.shape[0]):                         # <-- use k
            mz1 = float(q_arr[k, 0])
            mz_tolerance = _search_window_halfwidth_nb(mz1, float(cfg["tol"]), bool(cfg["use_ppm"]))
            lo_mz = mz1 - mz_tolerance
            hi_mz = mz1 + mz_tolerance
            prod_min[k] = np.searchsorted(lib.peaks_mz, lo_mz, side='left')
            prod_max[k] = np.searchsorted(lib.peaks_mz, hi_mz, side='right')
    else:
        prod_min = np.empty(0, dtype=np.int64)
        prod_max = np.empty(0, dtype=np.int64)

    q_pmz_val = float(q_pmz) if (q_pmz is not None) else np.nan

    _accumulate_nl_row_numba(scores,
                             q_arr[:, 0], q_arr[:, 1], q_pmz_val,
                             lib.nl_mz, lib.nl_int, lib.nl_spec_idx, lib.nl_product_idx,
                             lib.peaks_mz, lib.peaks_spec_idx,
                             float(cfg["tol"]), bool(cfg["use_ppm"]),
                             prefer_frag,
                             prod_min, prod_max)

    # identity mask
    if cfg["iden_tol"] is not None and q_pmz is not None:
        if cfg["iden_use_ppm"]:
            allow = np.abs(lib.precursor_mz - q_pmz) <= (cfg["iden_tol"] * 1e-6 * 0.5 * (lib.precursor_mz + q_pmz))
        else:
            allow = np.abs(lib.precursor_mz - q_pmz) <= cfg["iden_tol"]
        allow &= np.isfinite(lib.precursor_mz)
        scores[~allow] = 0.0

    return (row_idx, scores)


def _row_task_sparse(args):
    """Compute one row; return (row_index, cols, vals)."""
    row_idx, q_arr, q_pmz = args
    _, row = _row_task_dense((row_idx, q_arr, q_pmz))
    nz = np.nonzero(row)[0]
    return (row_idx, nz.astype(np.int64, copy=False), row[nz])
