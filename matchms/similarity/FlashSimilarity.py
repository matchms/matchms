import multiprocessing as mp
import platform
from typing import Dict, List, Optional, Tuple
import numpy as np
from numba import njit
from sparsestack import StackedSparseArray
from tqdm import tqdm
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .flash_utils import _build_library_index, _clean_and_weight, _within_tol


class FlashSimilarity(BaseSimilarity):
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
        Chose "hybrid" in combination with score_type="cosine" to compute the
        modified cosine score.
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
        Data type for the output scores. Default is np.float64 which properly accounts
        for highest resolution MS/MS data (even far beyond current MS/MS possibilties!).
        To save memory, np.float32 can be used instead, which is sufficient for peak 
        resolutions up to about 8,000,000.
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
                 dtype: np.dtype = np.float64):
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
        Uses the same preprocessing and scoring logic as the matrix path, but builds a tiny
        1-spectrum library from the query.
        
        Careful: This is not the fast intended use; better .matrix() instead.
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
    
        cfg = dict(
            tol=float(self.tolerance),
            use_ppm=bool(self.use_ppm),
            matching_mode=self.matching_mode,
            compute_nl=compute_nl,
            iden_tol=(None if self.identity_precursor_tolerance is None else float(self.identity_precursor_tolerance)),
            iden_use_ppm=bool(self.identity_use_ppm),
        )
        _set_globals(lib, cfg)

        worker = _row_task_entropy if self.score_type == "spectral_entropy" else _row_task_cosine
        _, row = worker((0, peaks_1, pmz_1))
        return np.asarray(row[0], dtype=self.dtype)

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
            print("FlashSimilarity.matrix: n_jobs != 1 is not yet implemented on Windows; "
                  "falling back to n_jobs=1.")
            n_jobs = 1

        # 1) Preprocess LIBRARY once
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
                                        weighing_type=("entropy" if self.score_type == "spectral_entropy"\
                                                       else "cosine"),
                                        dtype=self.dtype)
            lib_proc.append(cleaned)
            lib_pmz.append(None if pmz is None else float(pmz))

        compute_nl = (self.matching_mode in ("neutral_loss", "hybrid"))
        compute_l2 = (self.score_type == "cosine")

        lib = _build_library_index(
            lib_proc, lib_pmz,
            compute_neutral_loss=compute_nl,
            compute_l2_norm=compute_l2,
            dtype=self.dtype
        )

        # 2) Prepare output
        out = np.zeros((n_rows, n_cols), dtype=self.dtype)

        # 3) Prepare rows (references)
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
        _set_globals(lib, cfg)
        worker = _row_task_entropy if self.score_type == "spectral_entropy" else _row_task_cosine


        # 5) run — sequential or parallel
        if n_jobs in (None, 1, 0):
            iterator = row_inputs
            for item in tqdm(iterator, total=n_rows, desc="Flash entropy (matrix)"):
                row_idx, row = worker(item)
                out[row_idx, :] = row
        else:
            # Attempt Unix fork-based parallelism (not available on Windows; we already guarded above)
            n_cpus = mp.cpu_count()
            if n_jobs < 0:
                n_jobs = max(1, n_cpus + 1 + n_jobs)  # e.g., -1 -> n_cpus-1
            n_jobs = max(1, min(n_jobs, n_cpus))

            start_methods = mp.get_all_start_methods()
            use_fork = ("fork" in start_methods) and (platform.system() != "Windows")

            if use_fork:
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=n_jobs) as pool:
                    for result in tqdm(pool.imap(worker, row_inputs, chunksize=8),
                                    total=n_rows, desc=f"Flash entropy (parallel x{n_jobs})"):
                        i, row = result
                        out[i, :] = row
            else:
                # If fork is not available (e.g., certain environments), fall back to sequential with a note.
                print("FlashSimilarity.matrix: parallel execution requires 'fork'; "
                    "falling back to n_jobs=1.")
                for item in tqdm(row_inputs, total=n_rows, desc="Flash entropy (matrix)"):
                    i, row = worker(item)
                    out[i, :] = row

        if array_type == "numpy":
            return out
        elif array_type == "sparse":
            scores_array = StackedSparseArray(n_rows, n_cols)
            scores_array.add_dense_matrix(out.astype(self.score_datatype), "")
            return scores_array
        raise NotImplementedError("Output array type is unknown or not yet implemented.")


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


# ====================== Numba-accelerated accumulators (entropy) ======================

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


def _row_task_entropy(args):
    """
    Compute one matrix row for a given reference spectrum.

    Parameters
    ----------
    args : tuple
        (row_index, peaks_array, precursor_mz_or_None)

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
    _, row = _row_task_cosine((row_idx, q_arr, q_pmz))
    nz = np.nonzero(row)[0]
    return (row_idx, nz.astype(np.int64, copy=False), row[nz])


# ===================== Cosine related =====================

@njit(cache=True, nogil=True)
def _within_tol_nb(m1: float, m2: float, tol: float, use_ppm: bool) -> bool:
    """
    Numba-friendly symmetric tolerance check.

    Symmetric ppm definition:
        |m2 - m1| <= tol[ppm] * 1e-6 * 0.5 * (m1 + m2)
    Otherwise absolute Da tolerance:
        |m2 - m1| <= tol
    """
    if use_ppm:
        return abs(m2 - m1) <= (tol * 1e-6) * 0.5 * (m1 + m2)
    else:
        return abs(m2 - m1) <= tol


@njit(cache=True, nogil=True)
def _count_candidates_per_col_nb(query_mz: np.ndarray,
                                 query_int: np.ndarray,
                                 has_pmz: bool,
                                 query_pmz: float,
                                 peaks_mz: np.ndarray, peaks_spec_idx: np.ndarray,
                                 nl_mz: np.ndarray, nl_spec_idx: np.ndarray, nl_prod_idx: np.ndarray,
                                 tol: float, use_ppm: bool,
                                 do_frag: bool, do_nl: bool,
                                 n_cols: int) -> np.ndarray:
    """
    Count (fragment + neutral loss) candidate pairs per column (library spectrum).

    Returns
    -------
    counts_by_col : int32[n_cols]
        Number of candidate (i,j) pairs that belong to each library spectrum (column).
    """
    counts = np.zeros(n_cols, dtype=np.int32)

    # fragment matches
    if do_frag:
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            mz = float(query_mz[i])
            mz_tolerance = _search_window_halfwidth_nb(mz, tol, use_ppm)
            lo_mz = mz - mz_tolerance
            hi_mz = mz + mz_tolerance
            a = np.searchsorted(peaks_mz, lo_mz, side='left')
            b = np.searchsorted(peaks_mz, hi_mz, side='right')
            for j in range(a, b):
                # exact symmetric check to trim the searchsorted window
                if _within_tol_nb(mz, float(peaks_mz[j]), tol, use_ppm):
                    col = int(peaks_spec_idx[j])
                    counts[col] += 1

    # neutral-loss matches
    if do_nl and has_pmz and (nl_mz.size > 0):
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            loss = query_pmz - float(query_mz[i])
            mz_tolerance = _search_window_halfwidth_nb(mz, tol, use_ppm)
            lo_mz = mz - mz_tolerance
            hi_mz = mz + mz_tolerance
            a = np.searchsorted(nl_mz, lo_mz, side='left')
            b = np.searchsorted(nl_mz, hi_mz, side='right')
            for k in range(a, b):
                if _within_tol_nb(loss, float(nl_mz[k]), tol, use_ppm):
                    col = int(nl_spec_idx[k])
                    counts[col] += 1

    return counts


@njit(cache=True, nogil=True)
def _fill_candidates_per_col_nb(query_mz: np.ndarray,
                                query_int: np.ndarray,
                                has_pmz: bool, qpmz: float,
                                peaks_mz: np.ndarray, peaks_int: np.ndarray, peaks_spec_idx: np.ndarray,
                                nl_mz: np.ndarray, nl_spec_idx: np.ndarray, nl_prod_idx: np.ndarray,
                                tol: float, use_ppm: bool,
                                do_frag: bool, do_nl: bool,
                                col_offsets: np.ndarray,  # int64, length n_cols+1
                                counts_by_col: np.ndarray  # int64, length n_cols
                                ) -> tuple:
    """
    Materialize all candidates in CSR-like form grouped by column.

    Parameters
    ----------
    col_offsets : int64[n_cols+1]
        Prefix sums of counts (0, c0, c0+c1, ...).
    counts_by_col : int64[n_cols]
        As returned by _count_candidates_per_col_nb.

    Returns
    -------
    ref_idx  : int32[n_total]
    lib_idx  : int32[n_total]
    score    : float64[n_total]
    is_frag  : uint8[n_total]   (1 if fragment, 0 if NL)
    """
    n_cols = counts_by_col.shape[0]
    n_total = int(col_offsets[n_cols])

    ref_idx = np.empty(n_total, dtype=np.int32)
    lib_idx = np.empty(n_total, dtype=np.int32)
    score   = np.empty(n_total, dtype=np.float64)
    is_frag = np.empty(n_total, dtype=np.uint8)

    # running write cursors per column
    pos = col_offsets[:-1].copy()  # int64[n_cols]

    if do_frag:
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            mz = float(query_mz[i])
            mz_tolerance = _search_window_halfwidth_nb(m, tol, use_ppm)
            lo_mz = mz - mz_tolerance
            hi_mz = mz + mz_tolerance
            a = np.searchsorted(peaks_mz, lo_mz, side='left')
            b = np.searchsorted(peaks_mz, hi_mz, side='right')
            for j in range(a, b):
                mj = float(peaks_mz[j])
                if not _within_tol_nb(mz, mj, tol, use_ppm):
                    continue

                col = int(peaks_spec_idx[j])
                p = int(pos[col])
                ref_idx[p] = i
                lib_idx[p] = j
                score[p]   = Iq * float(peaks_int[j])
                is_frag[p] = 1
                pos[col] = p + 1

    if do_nl and has_pmz and (nl_mz.size > 0):
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            loss = qpmz - float(query_mz[i])
            mz_tolerance = _search_window_halfwidth_nb(loss, tol, use_ppm)
            lo_mz = loss - mz_tolerance
            hi_mz = loss + mz_tolerance
            a = np.searchsorted(nl_mz, lo_mz, side='left')
            b = np.searchsorted(nl_mz, hi_mz, side='right')
            for k in range(a, b):
                mk = float(nl_mz[k])
                if not _within_tol_nb(loss, mk, tol, use_ppm):
                    continue

                j = int(nl_prod_idx[k])        # product-peak index in global arrays
                col = int(nl_spec_idx[k])
                p = int(pos[col])
                ref_idx[p] = i
                lib_idx[p] = j
                score[p]   = Iq * float(peaks_int[j])
                is_frag[p] = 0
                pos[col] = p + 1

    return ref_idx, lib_idx, score, is_frag


@njit(cache=True, nogil=True)
def _greedy_scores_all_cols_nb(n_cols: int,
                               n_q: int,
                               col_offsets: np.ndarray,        # int64[n_cols+1]
                               ref_idx: np.ndarray,            # int32[n_total]
                               lib_idx: np.ndarray,            # int32[n_total]
                               score: np.ndarray,              # float64[n_total]
                               is_frag: np.ndarray,            # uint8[n_total]
                               lib_spec_l2: np.ndarray,        # float64[n_cols]
                               q_l2: float) -> np.ndarray:
    """
    Greedy, non-overlapping selection *per column* with score tie-break by fragment flag.

    For each column c:
      1) Build its candidate view: indices [start:end)
      2) Sort by key = score + (is_frag * eps), descending
      3) Walk candidates; accept if neither query-peak i nor lib-peak j used yet
      4) Accumulate dot and divide by norms

    Returns
    -------
    out : float64[n_cols]
        Cosine or modified-cosine per library spectrum.
    """
    out = np.zeros(n_cols, dtype=np.float64)
    # Query-peak usage flags (small; re-init per column is cheap)
    used_q = np.empty(n_q, dtype=np.uint8)

    eps = 1e-12  # tie-break to prefer fragment when scores are equal

    for col in range(n_cols):
        start = int(col_offsets[col])
        end   = int(col_offsets[col + 1])
        size  = end - start
        if size <= 0:
            continue

        # local views
        s_ref = ref_idx[start:end]
        s_lib = lib_idx[start:end]
        s_sc  = score[start:end]
        s_fg  = is_frag[start:end]

        # key for descending sort (score primary; fragment wins only on exact ties)
        key = np.empty(size, dtype=np.float64)
        for t in range(size):
            key[t] = s_sc[t] + (eps if s_fg[t] == 1 else 0.0)
        order = np.argsort(-key)  # descending

        # clear query usage (n_q is small)
        for qi in range(n_q):
            used_q[qi] = 0

        # track used library *global* j indices for this column only
        # small dynamic array; at most min(n_q, #uniq j in column)
        used_lib_j = np.empty(size, dtype=np.int32)
        used_lib_n = 0

        dot = 0.0
        for r in range(size):
            idx = int(order[r])
            i = int(s_ref[idx])
            j = int(s_lib[idx])

            if used_q[i] == 1:
                continue

            # linear membership test over selected lib j's; K remains small
            seen = False
            for u in range(used_lib_n):
                if used_lib_j[u] == j:
                    seen = True
                    break
            if seen:
                continue

            # accept
            used_q[i] = 1
            used_lib_j[used_lib_n] = j
            used_lib_n += 1
            dot += float(s_sc[idx])

        denom = q_l2 * float(lib_spec_l2[col])
        if denom > 0.0 and dot > 0.0:
            out[col] = dot / denom

    return out


# ==================== Row worker (cosine) ====================

def _row_task_cosine(args):
    """
    Compute one row (reference spectrum vs all library spectra) of
    (modified) cosine scores using a fully Numba-accelerated pipeline.

    Steps (per row / reference spectrum)
    ------------------------------------
    1. Build candidate (i,j) pairs against the global, sorted library index:
         - fragment matches around each reference m/z
         - neutral-loss matches around (precursor_ref - m/z), if enabled
    2. Group candidates by target library spectrum (column) in CSR-like form
       using prefix sums (`col_offsets`).
    3. For each column, greedily select a maximum-weight, non-overlapping
       subset of pairs (no query or library peak may be used twice),
       sorting by score (descending) with a tie-break to prefer fragments.
    4. Normalize by L2 norms to produce cosine / modified cosine.

    Notes
    -----
    * This path does **no global hybrid pruning**, matching your original design;
      the only hybrid rule is "fragment wins on ties for the same (i, j)".
    * Identity-gating is applied after scoring (unchanged).

    Parameters
    ----------
    args : tuple
        (row_index, peaks_array, precursor_mz_or_None)
    """
    row_idx, q_arr, q_pmz = args
    lib = _G_LIB
    cfg = _G_CFG

    if q_arr.size == 0:
        return (row_idx, np.zeros(lib.n_specs, dtype=lib.dtype))

    # reference (row) peaks
    q_mz  = q_arr[:, 0].astype(np.float64, copy=False)
    q_int = q_arr[:, 1].astype(np.float64, copy=False)

    # L2 norm of the reference
    q_l2 = float(np.sqrt(np.sum(q_int * q_int, dtype=np.float64)))
    if q_l2 == 0.0:
        return (row_idx, np.zeros(lib.n_specs, dtype=lib.dtype))

    # matching mode flags
    match_mode = cfg["matching_mode"]
    do_frag = (match_mode == "fragment") or (match_mode == "hybrid")
    do_nl   = (match_mode == "neutral_loss") or (match_mode == "hybrid")

    # neutral-loss only if library has NL view and we have a precursor on the row
    has_pmz = (q_pmz is not None)
    qpmz = float(q_pmz) if has_pmz else 0.0

    tol = float(cfg["tol"])
    use_ppm = bool(cfg["use_ppm"])

    n_cols = int(lib.n_specs)

    # 1) count per-column candidates (fragment + NL)
    counts_by_col = _count_candidates_per_col_nb(
        q_mz, q_int, has_pmz, qpmz,
        lib.peaks_mz.astype(np.float64, copy=False),
        lib.peaks_spec_idx.astype(np.int32, copy=False),
        (lib.nl_mz.astype(np.float64, copy=False) if (cfg["compute_nl"] and lib.nl_mz is not None) else np.empty(0, np.float64)),
        (lib.nl_spec_idx.astype(np.int32, copy=False) if (cfg["compute_nl"] and lib.nl_spec_idx is not None) else np.empty(0, np.int32)),
        (lib.nl_product_idx.astype(np.int32, copy=False) if (cfg["compute_nl"] and lib.nl_product_idx is not None) else np.empty(0, np.int32)),
        tol, use_ppm, do_frag, (do_nl and cfg["compute_nl"]), n_cols
    )

    # 2) prefix-sum to CSR offsets
    col_offsets = np.empty(n_cols + 1, dtype=np.int64)
    col_offsets[0] = 0
    for c in range(n_cols):
        col_offsets[c + 1] = col_offsets[c] + counts_by_col[c]

    n_total = int(col_offsets[-1])
    if n_total == 0:
        out = np.zeros(n_cols, dtype=lib.dtype)
    else:
        # 3) materialize candidates into CSR-like arrays
        ref_idx, lib_idx, score, is_frag = _fill_candidates_per_col_nb(
            q_mz, q_int,
            has_pmz, qpmz,
            lib.peaks_mz.astype(np.float64, copy=False),
            lib.peaks_int.astype(np.float64, copy=False),
            lib.peaks_spec_idx.astype(np.int32, copy=False),
            (lib.nl_mz.astype(np.float64, copy=False) if (cfg["compute_nl"] and lib.nl_mz is not None) else np.empty(0, np.float64)),
            (lib.nl_spec_idx.astype(np.int32, copy=False) if (cfg["compute_nl"] and lib.nl_spec_idx is not None) else np.empty(0, np.int32)),
            (lib.nl_product_idx.astype(np.int32, copy=False) if (cfg["compute_nl"] and lib.nl_product_idx is not None) else np.empty(0, np.int32)),
            tol, use_ppm, do_frag, (do_nl and cfg["compute_nl"]),
            col_offsets, counts_by_col
        )

        # 4) greedy select per column and normalize
        out64 = _greedy_scores_all_cols_nb(
            n_cols, q_mz.shape[0],
            col_offsets,
            ref_idx, lib_idx, score, is_frag,
            lib.spec_l2.astype(np.float64, copy=False),
            q_l2
        )
        out = out64.astype(lib.dtype, copy=False)

    # 5) optional identity gate (unchanged)
    iden_tol = cfg["iden_tol"]
    if (iden_tol is not None) and (q_pmz is not None):
        if cfg["iden_use_ppm"]:
            allow = np.abs(lib.precursor_mz - q_pmz) <= (iden_tol * 1e-6 * 0.5 * (lib.precursor_mz + q_pmz))
        else:
            allow = np.abs(lib.precursor_mz - q_pmz) <= iden_tol
        allow &= np.isfinite(lib.precursor_mz)
        out[~allow] = 0.0

    return (row_idx, out)
