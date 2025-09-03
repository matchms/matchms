import multiprocessing as mp
import platform
from typing import List, Optional, Tuple
import numpy as np
from numba import njit
from sparsestack import StackedSparseArray
from tqdm import tqdm
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .flash_utils import (
    _build_library_index,
    _clean_and_weight,
    _LibraryIndex,
)


class FlashSpectralEntropy(BaseSimilarity):
    """
    Flash entropy similarity (Li & Fiehn, 2023) with a fast .matrix() that
    builds a library-wide index over 'queries' and streams all 'references'
    through it.

    Key options:
      - matching_mode: 'fragment', 'neutral_loss', or 'hybrid' (fragment-priority).
      - tolerance in Da or symmetric ppm (use_ppm=True).
      - cleanup: remove precursor & > (precursor_mz - 1.6), 1% noise removal,
                 entropy weighting, normalize ∑I' = 0.5, optional within-peak merge.

    Notes:
      - .pair() works but is not the fast path. Use .matrix().
      - For identity-search behavior, pass identity_precursor_tolerance (Da or ppm).
    
    Parameters
    ----------
    score_type:
        Score type: 'spectral_entropy' (default) or 'cosine'.
    matching_mode:
        Matching mode: 'fragment', 'neutral_loss', or 'hybrid' (default is 'fragment').
        Chose "hybrid" in combination with score_type="cosine" to approximate
        the modified cosine score.
    tolerance:
        Matching tolerance in Da or ppm (use_ppm=True). Default is 0.02.
    use_ppm:
        If True, interpret `tolerance` as parts-per-million. Default is False.
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
                 score_type: str = "spectral_entropy",      # 'spectral_entropy' | 'cosine'
                 matching_mode: str = "fragment",           # 'fragment' | 'neutral_loss' | 'hybrid'
                 tolerance: float = 0.02,
                 use_ppm: bool = False,
                 remove_precursor: bool = True,
                 precursor_window: float = 1.6,
                 noise_cutoff: float = 0.01,
                 normalize_to_half: bool = True,
                 merge_within: float = 0.05,
                 identity_precursor_tolerance: Optional[float] = None,
                 identity_use_ppm: bool = False,
                 dtype: np.dtype = np.float32):
        if score_type not in ("spectral_entropy", "cosine"):
            raise ValueError("score_type must be 'spectral_entropy' or 'cosine'")
        if matching_mode not in ("fragment", "neutral_loss", "hybrid"):
            raise ValueError("mode must be 'fragment', 'neutral_loss', or 'hybrid'")
        self.score_type = score_type
        self.matching_mode = matching_mode
        self.tolerance = tolerance
        self.use_ppm = use_ppm
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
        peaks = spectrum.peaks.to_numpy
        pmz = spectrum.metadata.get("precursor_mz", None)
        cleaned = _clean_and_weight(peaks, pmz,
                                    remove_precursor=self.remove_precursor,
                                    precursor_window=self.precursor_window,
                                    noise_cutoff=self.noise_cutoff,
                                    normalize_to_half=self.normalize_to_half,
                                    merge_within_da=self.merge_within,
                                    weighing_type=("entropy" if self.score_type == "spectral_entropy" else "cosine"),
                                    dtype=self.dtype)
        return cleaned, (None if pmz is None else float(pmz))

    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
        """
        Compute Flash similarity for a single (reference, query) pair.
        Uses the same preprocessing and scoring logic as the matrix path, but
        builds a tiny 1-spectrum library from the query.
        """
        # preprocess both spectra
        peaks_1, pmz_1 = self._prepare(reference)
        peaks_2, pmz_2 = self._prepare(query)
        if peaks_1.size == 0 or peaks_2.size == 0:
            return np.asarray(0.0, dtype=self.dtype)
    
        # build 1-spec library index from the query
        compute_nl = (self.matching_mode in ("neutral_loss", "hybrid"))
        compute_l2 = (self.score_type == "cosine")

        lib = _build_library_index(
            [peaks_2], [pmz_2],
            compute_neutral_loss=compute_nl,
            compute_l2_norm=compute_l2,
            dtype=self.dtype
        )
    
        # compute scores into a 1-length buffer (single library spectrum)
        scores = np.zeros(1, dtype=self.dtype)
    
        # fragment path
        _accumulate_fragment_row_numba(
            scores,
            peaks_1[:, 0].astype(self.dtype, copy=False),
            peaks_1[:, 1].astype(self.dtype, copy=False),
            lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
            float(self.tolerance), bool(self.use_ppm)
        )
    
        # neutral-loss / hybrid path (if enabled & query has precursor)
        if self.matching_mode in ("neutral_loss", "hybrid") and (pmz_1 is not None):
            prefer_frag = (self.matching_mode == "hybrid")
    
            if prefer_frag:
                # precompute product-peak windows for ALL reference peaks
                prod_min = np.empty(peaks_1.shape[0], dtype=np.int64)
                prod_max = np.empty(peaks_1.shape[0], dtype=np.int64)
                for k in range(peaks_1.shape[0]):
                    mz1 = float(peaks_1[k, 0])
                    mz_tolerance = _search_window_halfwidth_nb(mz1, float(self.tolerance), bool(self.use_ppm))
                    lo_mz = mz1 - mz_tolerance
                    hi_mz = mz1 + mz_tolerance
                    prod_min[k] = np.searchsorted(lib.peaks_mz, lo_mz, side='left')
                    prod_max[k] = np.searchsorted(lib.peaks_mz, hi_mz, side='right')
            else:
                prod_min = np.empty(0, dtype=np.int64)
                prod_max = np.empty(0, dtype=np.int64)
    
            q_pmz_val = float(pmz_1) if (pmz_1 is not None) else np.nan
    
            _accumulate_nl_row_numba(
                scores,
                peaks_1[:, 0].astype(self.dtype, copy=False),
                peaks_1[:, 1].astype(self.dtype, copy=False),
                q_pmz_val,
                lib.nl_mz, lib.nl_int, lib.nl_spec_idx, lib.nl_product_idx,
                lib.peaks_mz, lib.peaks_spec_idx,
                float(self.tolerance), bool(self.use_ppm),
                prefer_frag,
                prod_min, prod_max
            )
    
        # identity gate (optional)
        if (self.identity_precursor_tolerance is not None) and (pmz_1 is not None):
            tol = float(self.identity_precursor_tolerance)
            lib_pmz = float(lib.precursor_mz[0])
            if self.identity_use_ppm:
                allow = abs(lib_pmz - pmz_1) <= (tol * 1e-6 * 0.5 * (lib_pmz + pmz_1))
            else:
                allow = abs(lib_pmz - pmz_1) <= tol
            if not (allow and np.isfinite(lib_pmz)):
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

        # ---- Windows safety fallback ----
        if platform.system() == "Windows" and n_jobs not in (None, 1, 0):
            print("FlashSpectralEntropy.matrix: n_jobs != 1 is not yet implemented on Windows; "
                  "falling back to n_jobs=1.")
            n_jobs = 1

        # 1) preprocess LIBRARY once
        lib_proc = []
        lib_pmz = []
        for s in queries:
            peaks = s.peaks.to_numpy
            pmz = s.metadata.get("precursor_mz", None)
            cleaned = _clean_and_weight(peaks, pmz,
                                        remove_precursor=self.remove_precursor,
                                        precursor_window=self.precursor_window,
                                        noise_cutoff=self.noise_cutoff,
                                        normalize_to_half=self.normalize_to_half,
                                        merge_within_da=self.merge_within,
                                        weighing_type="entropy",
                                        dtype=self.dtype)
            lib_proc.append(cleaned)
            lib_pmz.append(None if pmz is None else float(pmz))

        compute_nl = (self.matching_mode in ("neutral_loss", "hybrid"))
        lib = _build_library_index(lib_proc, lib_pmz, compute_neutral_loss=compute_nl, dtype=self.dtype)

        # 2) prepare output
        out = np.zeros((n_rows, n_cols), dtype=self.dtype)

        # 3)  preprocess each reference spectrum
        row_inputs = []
        for i, ref in enumerate(references):
            peaks_r, pmz_r = self._prepare(ref)
            row_inputs.append((i, peaks_r, pmz_r))

        # 4) configuration passed to workers
        cfg = dict(
            tol=float(self.tolerance),
            use_ppm=bool(self.use_ppm),
            matching_mode=self.matching_mode,
            compute_nl=compute_nl,
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


# ====================== Numba-accelerated accumulators ======================-

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

        mz_tolerance = _search_window_halfwidth_nb(mz_q, tol, use_ppm)
        lo_mz = mz_q - mz_tolerance
        hi_mz = mz_q + mz_tolerance

        a = np.searchsorted(lib_mz, lo_mz, side='left')
        b = np.searchsorted(lib_mz, hi_mz, side='right')
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
    Accumulate neutral-loss (NL) contributions for a single row (one reference spectrum).

    For each reference peak of m/z = m_ref with intensity Iq:
      - Compute the loss L = precursor_ref - m_ref.
      - Find all library neutral-loss peaks within tolerance of L using nl_mz (sorted).
      - For each candidate:
          * If hybrid mode (prefer_fragments=True), apply two pruning rules:
              RULE 1: Drop if the candidate's corresponding product-peak index
                      lies inside ANY fragment window of the current reference spectrum.
              RULE 2: Drop if the current reference *fragment* also matches the same
                      library spectrum (i.e., we prefer fragment matches over NL matches).
          * Add the entropy increment to scores[col].

    Arrays
    ------
    nl_* arrays are built from the library (query) spectra with known precursors.
    nl_product_pos maps each NL entry back to a position in the global product-peak
    arrays; this is used by hybrid rules.
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

        # (Hybrid) fragment window for this reference peak
        ap = 0
        bp = 0
        if prefer_fragments:
            mz1 = float(q_mz[i])
            mz_tolerance = _search_window_halfwidth_nb(mz1, tol, use_ppm)
            lo_mz = mz1 - mz_tolerance
            hi_mz = mz1 + mz_tolerance
            ap = np.searchsorted(peaks_mz, lo_mz, side='left')
            bp = np.searchsorted(peaks_mz, hi_mz, side='right')

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

            if prefer_fragments:
                # RULE 1: drop if product peak falls in ANY fragment window of the query spectrum
                if _in_any_fragment_window(int(nl_prod_idx[j]), prod_min, prod_max):
                    continue

                # RULE 2 (per-peak): if this query peak also fragment-matches the same library spectrum, drop
                if ap < bp and _spec_in_fragment_window(col, peaks_spec, ap, bp):
                    continue

            incr= _xlog2_scalar_nb(lib + Iq) - _xlog2_scalar_nb(Iq) - _xlog2_scalar_nb(lib)
            scores[col] += incr



# ===================== worker plumbing =====================

# Globals visible to workers
_G_LIB = None
_G_CFG = None

def _set_globals(lib_obj, cfg):
    """
    Install the shared library index and constant config in module-level globals.

    Worker functions read from these globals to avoid re-pickling large arrays
    for every work item (especially efficient with fork on Unix).
    """
    global _G_LIB, _G_CFG
    _G_LIB = lib_obj
    _G_CFG = cfg

def _row_task_dense(args):
    """
    Compute one matrix row for a given reference spectrum.

    Returns
    -------
    (row_index, row_scores)
        `row_scores` is a dense vector of length n_library_spectra in the
        same float dtype as the index. This function is safe to call from
        a process pool since it only reads globals set by `_set_globals`.
    """
    row_idx, q_arr, q_pmz = args
    cfg = _G_CFG
    lib = _G_LIB
    dtype = lib.dtype
    scores = np.zeros(lib.n_specs, dtype=dtype)

    _accumulate_fragment_row_numba(scores,
                                   q_arr[:, 0], q_arr[:, 1],
                                   lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
                                   float(cfg["tol"]), bool(cfg["use_ppm"]))

    # neutral-loss / hybrid ONLY if library has NL view
    if cfg["compute_nl"]:
        prefer_frag = (cfg["matching_mode"] == "hybrid")
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
