import multiprocessing as mp
import platform
from multiprocessing.shared_memory import SharedMemory
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
        # Convenience, still float32; for speed use .matrix()
        A, pmzA = self._prepare(reference)
        B, pmzB = self._prepare(query)
        if A.size == 0 or B.size == 0:
            return np.asarray(0.0, dtype=self.dtype)

        # Build tiny "lib" from B and run the row code
        lib = _LibraryIndex(self.dtype)
        lib.n_specs = 1
        lib.peaks_mz = B[:, 0]
        lib.peaks_int = B[:, 1]
        lib.peaks_spec_idx = np.zeros(B.shape[0], dtype=np.int32)
        if self.mode in ("neutral_loss", "hybrid") and pmzB is not None:
            lib.nl_mz = (pmzB - B[:, 0]).astype(self.dtype, copy=False)
            order = np.argsort(lib.nl_mz)
            lib.nl_mz = lib.nl_mz[order]
            lib.nl_int = B[:, 1][order]
            lib.nl_spec_idx = np.zeros(B.shape[0], dtype=np.int32)
            lib.nl_product_idx = order.astype(np.int64, copy=False)
        else:
            lib.nl_mz = lib.nl_int = lib.nl_spec_idx = lib.nl_product_idx = np.zeros(0, dtype=self.dtype)
        lib.precursor_mz = np.array([pmzB if pmzB is not None else np.nan], dtype=self.dtype)

        scores = np.zeros(1, dtype=self.dtype)
        _accumulate_fragment_row(scores, A[:, 0], A[:, 1],
                                 lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
                                 self.tolerance, self.use_ppm, self.dtype)
        if self.mode in ("neutral_loss", "hybrid"):
            _accumulate_nl_row(scores, A[:, 0], A[:, 1], pmzA,
                               lib.nl_mz, lib.nl_int, lib.nl_spec_idx, lib.nl_product_idx,
                               lib.peaks_mz, lib.peaks_spec_idx,
                               self.tolerance, self.use_ppm, self.dtype,
                               prefer_fragments=(self.mode == "hybrid"))
        # identity gate
        if self.identity_precursor_tolerance is not None and pmzA is not None:
            if self.identity_use_ppm:
                allow = abs(lib.precursor_mz[0] - pmzA) <= (
                    self.identity_precursor_tolerance * 1e-6 * 0.5 * (lib.precursor_mz[0] + pmzA)
                )
            else:
                allow = abs(lib.precursor_mz[0] - pmzA) <= self.identity_precursor_tolerance
            if not (allow and np.isfinite(lib.precursor_mz[0])):
                scores[0] = 0.0

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
                i, row = worker(item)
                out[i, :] = row
                
            if array_type == "numpy":
                return out
            elif array_type == "sparse":
                scores_array = StackedSparseArray(n_rows, n_cols)
                scores_array.add_dense_matrix(out.astype(self.score_datatype), "")
                return scores_array
            raise NotImplementedError("Output array type is unknown or not yet implemented.")

        # parallel
        n_cpus = mp.cpu_count()
        if n_jobs < 0:
            n_jobs = max(1, n_cpus + 1 + n_jobs)  # e.g., -1 -> n_cpus-1
        n_jobs = max(1, min(n_jobs, n_cpus))

        start_methods = mp.get_all_start_methods()
        use_fork = "fork" in start_methods and platform.system() != "Windows"

        # Strategy A (Unix): fork — globals are shared copy-on-write (zero copy)
        if use_fork:
            ctx = mp.get_context("fork")
            _set_globals(lib, cfg)  # make available pre-fork
            worker = _row_task_dense #  if array_type == "numpy" else _row_task_sparse (TODO?)
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

        # Strategy B (Windows / spawn): SharedMemory for big arrays
        def _to_shm(a: np.ndarray):
            if a is None or a.size == 0:
                return None
            shm = SharedMemory(create=True, size=a.nbytes)
            np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)[:] = a
            return dict(name=shm.name, shape=a.shape, dtype=str(a.dtype)), shm

        def _from_shm(meta):
            shm = SharedMemory(name=meta["name"])
            arr = np.ndarray(meta["shape"], dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
            return shm, arr

        # Pack library arrays into shared memory blocks
        shms = []
        metas = {}

        for key in ("peaks_mz", "peaks_int", "peaks_spec_idx",
                    "nl_mz", "nl_int", "nl_spec_idx", "nl_product_idx",
                    "precursor_mz"):
            arr = getattr(lib, key)
            meta, shm = (None, None)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                meta, shm = _to_shm(arr)
                shms.append(shm)
            metas[key] = meta

        worker = _row_task_dense #  if array_type == "numpy" else _row_task_sparse (TODO?)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs,
                      initializer=_init_spawn_worker,
                      initargs=(metas, cfg, str(self.dtype))) as pool:
            for result in tqdm(pool.imap(worker, row_inputs, chunksize=4),
                               total=n_rows, desc=f"Flash entropy (parallel x{n_jobs})"):
                i, row = result
                out[i, :] = row

        # parent cleanup
        for shm in shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

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
    if peaks.size == 0:
        return np.empty((0, 2), dtype=dtype)

    mz = _as_dtype(peaks[:, 0], dtype)
    inten = _as_dtype(peaks[:, 1], dtype)

    if remove_precursor and (precursor_mz is not None):
        mask = mz <= (precursor_mz - precursor_window)
        mz, inten = mz[mask], inten[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    if noise_cutoff and noise_cutoff > 0.0:
        #thr = dtype.type(mz.dtype.type(inten.max()) * noise_cutoff)
        thr = inten.max() * noise_cutoff
        mask = inten >= thr
        mz, inten = mz[mask], inten[mask]
        if mz.size == 0:
            return np.empty((0, 2), dtype=dtype)

    inten = _entropy_weight(inten, dtype)

    if merge_within_da and merge_within_da > 0.0 and mz.size > 1:
        peaks = _merge_within(np.column_stack((mz, inten)), merge_within_da, dtype)
        mz, inten = peaks[:, 0], peaks[:, 1]
    else:
        order = np.argsort(mz)
        mz, inten = mz[order], inten[order]

    s = float(inten.sum(dtype=np.float64))
    if s > 0.0 and normalize_to_half:
        inten = (inten * (0.5 / s)).astype(dtype, copy=False)

    return np.column_stack((mz, inten))


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


# ===================== accumulators (row) =====================

def _accumulate_fragment_row(scores: np.ndarray,
                             q_mz: np.ndarray, q_int: np.ndarray,
                             peaks_mz: np.ndarray, peaks_int: np.ndarray, peaks_spec: np.ndarray,
                             tol: float, use_ppm: bool, dtype: np.dtype):
    for i in range(q_mz.shape[0]):
        mz1 = float(q_mz[i])
        Iq = float(q_int[i])
        if Iq <= 0.0:
            continue
        mz_tolerance = _search_window_halfwidth(mz1, tol, use_ppm, dtype)
        lo_mz = mz1 - mz_tolerance
        hi_mz = mz1 + mz_tolerance
        a = np.searchsorted(peaks_mz, lo_mz, side='left')
        b = np.searchsorted(peaks_mz, hi_mz, side='right')
        if a >= b:
            continue
        # refine for symmetric ppm
        mz2 = peaks_mz[a:b].astype(dtype, copy=False)
        if use_ppm:
            mask = np.abs(mz2 - mz1) <= ((tol * 1e-6) * (0.5) * (mz2 + mz1))
        else:
            mask = np.abs(mz2 - mz1) <= tol
        if not np.any(mask):
            continue
        lib = peaks_int[a:b][mask].astype(dtype, copy=False)
        cols = peaks_spec[a:b][mask]
        v = _xlog2_vec(lib + Iq, dtype) - _xlog2_scalar(Iq, dtype) - _xlog2_vec(lib, dtype)
        np.add.at(scores, cols, v.astype(dtype, copy=False))

def _accumulate_nl_row(scores: np.ndarray,
                       q_mz: np.ndarray, q_int: np.ndarray, q_pmz: Optional[float],
                       nl_mz: np.ndarray, nl_int: np.ndarray, nl_spec: np.ndarray, nl_prod_idx: np.ndarray,
                       peaks_mz: np.ndarray, peaks_spec: np.ndarray,
                       tol: float, use_ppm: bool, dtype: np.dtype,
                       prefer_fragments: bool = False):
    if q_pmz is None or q_mz.size == 0 or nl_mz.size == 0:
        return

    # Precompute product ranges for each query peak (for hybrid priority)
    prod_min = np.empty(q_mz.shape[0], dtype=np.int64)
    prod_max = np.empty(q_mz.shape[0], dtype=np.int64)
    if prefer_fragments:
        for i in range(q_mz.shape[0]):
            mz1 = float(q_mz[i])
            mz_tolerance = _search_window_halfwidth(mz1, tol, use_ppm, dtype)
            lo_mz = mz1 - mz_tolerance
            hi_mz = mz1 + mz_tolerance
            prod_min[i] = np.searchsorted(peaks_mz, lo_mz, side='left')
            prod_max[i] = np.searchsorted(peaks_mz, hi_mz, side='right')

    for i in range(q_mz.shape[0]):
        Iq = float(q_int[i])
        if Iq <= 0.0:
            continue
        loss = float(q_pmz) - float(q_mz[i])
        mz_tolerance = _search_window_halfwidth(loss, tol, use_ppm, dtype)
        lo_mz = loss - mz_tolerance
        hi_mz = loss + mz_tolerance
        a = np.searchsorted(nl_mz, lo_mz, side='left')
        b = np.searchsorted(nl_mz, hi_mz, side='right')
        if a >= b:
            continue

        nl2 = nl_mz[a:b].astype(dtype, copy=False)
        if use_ppm:
            mask = np.abs(nl2 - loss) <= ((tol * 1e-6) * 0.5 * (nl2 + loss))
        else:
            mask = np.abs(nl2 - loss) <= tol
        if not np.any(mask):
            continue

        lib = nl_int[a:b][mask].astype(dtype, copy=False)
        cols = nl_spec[a:b][mask]
        prod_idx_slice = nl_prod_idx[a:b][mask]

        if prefer_fragments:
            # RULE 1: If NL refers to a product peak that lies in any fragment window of this query spectrum
            # (for any query peak), drop it.
            s1 = np.searchsorted(prod_min, prod_idx_slice, side='right')
            s2 = np.searchsorted(prod_max - 1, prod_idx_slice, side='left')
            drop = s1 > s2
            if np.any(drop):
                keep = ~drop
                lib = lib[keep]
                cols = cols[keep]
                prod_idx_slice = prod_idx_slice[keep]
                if lib.size == 0:
                    continue

            # RULE 2 (per-peak): if this query peak also fragment-matches the same library spectrum, drop NL
            mz1 = float(q_mz[i])
            mz_tolerance = _search_window_halfwidth(mz1, tol, use_ppm, dtype)
            lo_mz = mz1 - mz_tolerance
            hi_mz = mz1 + mz_tolerance
            ap = np.searchsorted(peaks_mz, lo_mz, side='left')
            bp = np.searchsorted(peaks_mz, hi_mz, side='right')
            if ap < bp:
                cols_frag = np.unique(peaks_spec[ap:bp])
                if cols_frag.size > 0:
                    mask2 = ~np.isin(cols, cols_frag, assume_unique=False)
                    if not np.all(mask2):
                        cols = cols[mask2]
                        lib = lib[mask2]
                        if lib.size == 0:
                            continue

        v = _xlog2_vec(lib + Iq, dtype) - _xlog2_scalar(Iq, dtype) - _xlog2_vec(lib, dtype)
        np.add.at(scores, cols, v.astype(dtype, copy=False))


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
    i, q_arr, q_pmz = args
    cfg = _G_CFG
    lib = _G_LIB
    dtype = lib.dtype
    scores = np.zeros(lib.n_specs, dtype=dtype)

    _accumulate_fragment_row(scores,
                             q_arr[:, 0], q_arr[:, 1],
                             lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
                             cfg["tol"], cfg["use_ppm"], dtype)

    if cfg["build_nl"]:
        _accumulate_nl_row(scores,
                           q_arr[:, 0], q_arr[:, 1], q_pmz,
                           lib.nl_mz, lib.nl_int, lib.nl_spec_idx, lib.nl_product_idx,
                           lib.peaks_mz, lib.peaks_spec_idx,
                           cfg["tol"], cfg["use_ppm"], dtype,
                           prefer_fragments=(cfg["mode"] == "hybrid"))

    # identity mask
    if cfg["iden_tol"] is not None and q_pmz is not None:
        if cfg["iden_use_ppm"]:
            allow = np.abs(lib.precursor_mz - q_pmz) <= (cfg["iden_tol"] * 1e-6 * 0.5 * (lib.precursor_mz + q_pmz))
        else:
            allow = np.abs(lib.precursor_mz - q_pmz) <= cfg["iden_tol"]
        allow &= np.isfinite(lib.precursor_mz)
        scores[~allow] = 0.0

    return (i, scores)

def _row_task_sparse(args):
    """Compute one row; return (row_index, cols, vals)."""
    i, q_arr, q_pmz = args
    _, row = _row_task_dense((i, q_arr, q_pmz))
    nz = np.nonzero(row)[0]
    return (i, nz.astype(np.int64, copy=False), row[nz])


def _init_spawn_worker(metas_, cfg_, dtype_str):
    """Picklable initializer for Windows spawn."""
    lib_local = _LibraryIndex(np.dtype(dtype_str))

    def _from_shm(meta):
        shm = SharedMemory(name=meta["name"])
        arr = np.ndarray(meta["shape"], dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
        return shm, arr

    for key, meta in metas_.items():
        if meta is None:
            if key in ("peaks_spec_idx", "nl_spec_idx"):
                arr = np.zeros(0, dtype=np.int32)
            elif key in ("nl_product_idx",):
                arr = np.zeros(0, dtype=np.int64)
            else:
                arr = np.zeros(0, dtype=lib_local.dtype)
            setattr(lib_local, key, arr)
        else:
            shm, arr = _from_shm(meta)
            setattr(lib_local, f"_{key}_shm", shm)  # keep handle alive
            setattr(lib_local, key, arr)

    _set_globals(lib_local, cfg_)
