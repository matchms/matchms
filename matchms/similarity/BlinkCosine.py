from typing import List, Optional, Tuple
import numpy as np
from numba import njit  # TODO: check if numba is necessary/useful here
from scipy.sparse import coo_array, csr_array
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


@njit(cache=True, fastmath=True)
def _windowed_sum_numba(source_bins: np.ndarray, source_vals: np.ndarray,
                        query_positions: np.ndarray, R: int) -> np.ndarray:
    """
    Two-pointer windowed sum for sorted integer arrays.

    For each position `pos` in `query_positions`, compute:
        sum(source_vals[j]) where source_bins[j] ∈ [pos - R, pos + R]

    Parameters
    ----------
    source_bins : np.ndarray
        Sorted 1D array of integer bin indices (ascending).
    source_vals : np.ndarray
        1D array of values aligned to `source_bins`.
    query_positions : np.ndarray
        1D array of integer bin positions at which to evaluate the windowed sum.
    R : int
        Window radius in bins (inclusive on both sides).

    Returns
    -------
    np.ndarray
        1D array of windowed sums, same length/order as `query_positions`.
    """
    n_bins = source_bins.size
    m = query_positions.size
    out = np.zeros(m, dtype=np.float32)
    if n_bins == 0 or m == 0:
        return out

    left_idx = 0
    right_idx = 0
    acc = 0.0

    for i in range(m):
        pos = query_positions[i]
        left_bound = pos - R
        right_bound = pos + R

        # expand right edge
        while right_idx < n_bins and source_bins[right_idx] <= right_bound:
            acc += source_vals[right_idx]
            right_idx += 1

        # contract left edge
        while left_idx < right_idx and source_bins[left_idx] < left_bound:
            acc -= source_vals[left_idx]
            left_idx += 1

        out[i] = acc
    return out


class BlinkCosine(BaseSimilarity):
    """
    BLINK-style approximate cosine similarity for mass spectra with fast `.pair()` and `.matrix()`.
    This score is implemented based on the method BLINK, proposed by Harwood et al. (2023,
    https://www.nature.com/articles/s41598-023-40496-9).

    * Integer binning with `bin_width` (Da); tolerance window is ± floor(tolerance/bin_width) bins.
    * Per-spectrum L2 normalization (after optional mz/intensity weighting).
    * Blur only one side (queries in `.matrix()`, smaller spectrum in `.pair()`).
    * Pairwise returns (score, ~matches). Matrix returns only scores.

    Parameters
    ----------
    tolerance:
        True m/z tolerance (Da). Peaks within +/- tolerance are considered matches. Default 0.01.
    bin_width:
        Discretization width (Da). Default 0.001 (1 mDa). Effective radius R=floor(tolerance/bin_width).
    mz_power:
        Power for mz weighting (intensity *= mz**mz_power). Default 0.0.
    intensity_power:
        Power for intensity weighting before normalization. Default 1.0 (set 0.5 for sqrt scaling).
    clip_to_one:
        Clip score to [0,1]. Default True.

    use_numba : bool
        Use numba-accelerated pairwise kernel when available. Default True.
    prefilter : bool
        Apply BLINK-like pre-filtering (remove <1% base peak, > precursor m/z, zeros). Default True.
    min_relative_intensity : float
        Relative base-peak threshold for prefilter. Default 0.01 (1%).
    crop_above_precursor : bool
        Drop fragments > precursor m/z if available in metadata. Default True.
    remove_zero_intensities : bool
        Remove peaks with intensity <= 0. Default True.
    top_k : Optional[int]
        Keep only top-K most intense fragments after other filters (per spectrum). Default None.

    # Batching (matrix path)
    batch_size : int
        Number of query spectra per batch in `.matrix()`. Default 1024.
    sparse_score_min : float
        When array_type='sparse', drop scores < sparse_score_min. Default 0.0.
    """

    is_commutative = True
    score_datatype = [("score", np.float32), ("matches", "int")]

    def __init__(
        self,
        tolerance: float = 0.01,
        bin_width: float = 0.001,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        clip_to_one: bool = True,
        # extras
        use_numba: bool = True,
        prefilter: bool = True,
        min_relative_intensity: float = 0.01,
        crop_above_precursor: bool = True,
        remove_zero_intensities: bool = True,
        top_k: Optional[int] = None,
        # batching
        batch_size: int = 1024,
        sparse_score_min: float = 0.0,
    ):
        self.tolerance = float(tolerance)
        self.bin_width = float(bin_width)
        self.mz_power = float(mz_power)
        self.intensity_power = float(intensity_power)
        self.clip_to_one = bool(clip_to_one)

        self.use_numba = bool(use_numba)
        self.prefilter = bool(prefilter)
        self.min_relative_intensity = float(min_relative_intensity)
        self.crop_above_precursor = bool(crop_above_precursor)
        self.remove_zero_intensities = bool(remove_zero_intensities)
        self.top_k = top_k if (top_k is None) else int(top_k)

        self.batch_size = int(batch_size)
        self.sparse_score_min = float(sparse_score_min)

        self._R = max(0, int(np.floor(self.tolerance / self.bin_width)))

    # --------------------------- Public API ---------------------------

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate BLINK-style cosine between two spectra.
        
        Parameters
        ----------
        reference:
            Single reference spectrum.
        query:
            Single query spectrum.
        """
        rbins, rvals, rcounts = self._prep_spectrum(reference)
        qbins, qvals, qcounts = self._prep_spectrum(query)

        if rbins.size == 0 or qbins.size == 0:
            return np.asarray((0.0, 0), dtype=self.score_datatype)

        # Blur smaller side, evaluate at the other's bins
        if qbins.size <= rbins.size:
            win = self._windowed_sum(qbins, qvals, rbins, self._R)
            score = float(np.dot(win, rvals))
            matches = int(self._windowed_sum(qbins, qcounts.astype(np.float32), rbins, self._R).sum())
        else:
            win = self._windowed_sum(rbins, rvals, qbins, self._R)
            score = float(np.dot(win, qvals))
            matches = int(self._windowed_sum(rbins, rcounts.astype(np.float32), qbins, self._R).sum())

        if self.clip_to_one:
            score = min(score, 1.0)
        return np.asarray((score, matches), dtype=self.score_datatype)

    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False):
        """
        All-vs-all BLINK-style cosine scores.

        Implementation:
        - Build a *global dense bin axis* in integer bins from min to max across refs+queries
          (rows ~ (max_bin - min_bin + 1)), which keeps matrices sparse.
        - Build a CSR intensity matrix for refs (rows=bins, cols=ref spectra) after per-spectrum L2 normalization.
        - For queries, build per-batch *blurred* CSR by expanding each nonzero to its ±R neighbors.
        - Multiply: scores_batch = (I_ref.T @ I_qry_blur), accumulate into the final output.

        Parameters
        ----------
        references:
            List of reference spectra.
        queries:
            List of query spectra.
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a COO-sparse array

        Returns
        -------
        numpy.ndarray or scipy.sparse.coo_array
            If array_type == 'numpy': dense (n_ref, n_query)
            If array_type == 'sparse': COO sparse (n_ref, n_query), dropping scores < sparse_score_min
        """
        if array_type not in ("numpy", "sparse"):
            raise ValueError("array_type must be 'numpy' or 'sparse'.")

        # Preprocess all spectra (bins, normalized intensity values)
        prepped_refs = [self._prep_spectrum(s) for s in references]
        prepped_qrys = [self._prep_spectrum(s) for s in queries]

        # Early exit if any side empty
        n_ref = len(prepped_refs)
        n_qry = len(prepped_qrys)
        if n_ref == 0 or n_qry == 0:
            if array_type == "numpy":
                return np.zeros((n_ref, n_qry), dtype=np.float32)
            return coo_array((n_ref, n_qry), dtype=np.float32)

        # Collect global bin range
        all_bins_list = [b for (b, _, _) in prepped_refs if b.size] + [b for (b, _, _) in prepped_qrys if b.size]
        if not all_bins_list:
            if array_type == "numpy":
                return np.zeros((n_ref, n_qry), dtype=np.float32)
            return coo_array((n_ref, n_qry), dtype=np.float32)

        global_min = min(int(b.min()) for b in all_bins_list)
        global_max = max(int(b.max()) for b in all_bins_list)
        n_rows = int(global_max - global_min + 1)
        offset = -global_min  # row_index = bin + offset

        # Build reference intensity CSR once
        I_ref = self._build_intensity_csr(prepped_refs, n_rows, offset)

        # Output container
        if array_type == "numpy":
            S = np.zeros((n_ref, n_qry), dtype=np.float32)
        else:
            sparse_rows = []
            sparse_cols = []
            sparse_data = []

        # Batch queries -> blur -> multiply
        j = 0
        while j < n_qry:
            j0 = j
            j1 = min(j + self.batch_size, n_qry)
            batch = prepped_qrys[j0:j1]
            IQ_blur = self._build_blurred_csr(batch, n_rows, offset, self._R)  # blurred queries (rows x B)

            # scores_batch = I_ref.T @ IQ_blur  -> shape (n_ref, B)
            scores_batch_sparse = I_ref.T @ IQ_blur

            if array_type == "numpy":
                S[:, j0:j1] = scores_batch_sparse.toarray()
            else:
                scores_batch_sparse = scores_batch_sparse.tocoo()
                # Apply threshold if requested
                if self.sparse_score_min > 0.0:
                    mask = np.abs(scores_batch_sparse.data) >= self.sparse_score_min
                else:
                    mask = np.ones_like(scores_batch_sparse.data, dtype=bool)
                sparse_rows.append(scores_batch_sparse.row[mask])
                sparse_cols.append(scores_batch_sparse.col[mask] + j0)
                sparse_data.append(scores_batch_sparse.data[mask])

            j = j1

        if array_type == "numpy":
            if self.clip_to_one:
                np.minimum(S, 1.0, out=S)
            if is_symmetric and n_ref == n_qry and references is queries:
                # Optional: enforce exact symmetry (no computational saving here)
                S = 0.5 * (S + S.T)
            return S
        else:
            if sparse_rows:
                rows = np.concatenate(sparse_rows) if len(sparse_rows) > 1 else sparse_rows[0]
                cols = np.concatenate(sparse_cols) if len(sparse_cols) > 1 else sparse_cols[0]
                data = np.concatenate(sparse_data) if len(sparse_data) > 1 else sparse_data[0]
            else:
                rows = np.array([], dtype=np.int32)
                cols = np.array([], dtype=np.int32)
                data = np.array([], dtype=np.float32)
            if self.clip_to_one and data.size:
                np.minimum(data, 1.0, out=data)
            return coo_array((data, (rows, cols)), shape=(n_ref, n_qry), dtype=np.float32)

    # --------------------------- Internal helpers ---------------------------

    def _prefilter_arrays(self, mz: np.ndarray, intens: np.ndarray, spectrum: SpectrumType):
        """
        Apply BLINK-like prefiltering on raw m/z and intensity arrays.

        Parameters
        ----------
        mz : np.ndarray
            1D array of m/z values.
        intens : np.ndarray
            1D array of intensities aligned with `mz`.
        spectrum : SpectrumType
            Spectrum object (used to read optional metadata like precursor m/z).
        """
        if mz.size == 0:
            return mz, intens

        # Relative intensity threshold
        if self.min_relative_intensity > 0.0:
            base = intens.max() if intens.size else 1.0
            thr = self.min_relative_intensity * base
            mask = intens >= thr
            mz = mz[mask]
            intens = intens[mask]
            if mz.size == 0:
                return mz, intens

        # Crop above precursor m/z
        if self.crop_above_precursor:
            prec = None
            # Try common metadata locations
            try:
                prec = spectrum.get("precursor_mz")
            except Exception:
                pass
            if prec is not None:
                mask = mz <= float(prec)
                mz = mz[mask]
                intens = intens[mask]

        # Keep top-K most intense if requested
        if (self.top_k is not None) and (mz.size > self.top_k):
            idx = np.argpartition(intens, -self.top_k)[-self.top_k:]
            idx.sort()
            mz = mz[idx]
            intens = intens[idx]

        return mz, intens

    def _prep_spectrum(self, spectrum: SpectrumType):
        """
        Prepare a spectrum for scoring.

        Steps
        -----
        1) Optional prefiltering
        2) Optional m/z and intensity weighting
        3) Integer binning to nearest bin (defined by `bin_width`)
        4) Aggregation of duplicate bins
        5) L2 normalization of binned intensities

        Parameters
        ----------
        spectrum : SpectrumType
            Spectrum object with `.peaks.mz` and `.peaks.intensities`.

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            Tuple of (unique_bins, normalized_values, counts_per_bin).
        """
        mz = spectrum.peaks.mz
        intens = spectrum.peaks.intensities

        if self.prefilter:
            mz, intens = self._prefilter_arrays(mz, intens, spectrum)

        if mz.size == 0:
            return (np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))

        # Optional weighting
        if self.mz_power != 0.0:
            intens = intens * np.power(mz, self.mz_power, dtype=np.float32)
        if self.intensity_power != 1.0:
            intens = np.power(intens, self.intensity_power, dtype=np.float32)

        # Bin to nearest integer bin
        mz_binned = np.floor(mz / self.bin_width + 0.5).astype(np.int32)

        # Aggregate duplicates (sorting not needed since mz values are sorted)
        uniq, idx, counts = np.unique(mz_binned, return_index=True, return_counts=True)
        intensity_sum = np.add.reduceat(intens, idx)

        # L2 normalize intensities (Sum of all intensities == 1)
        norm = np.linalg.norm(intensity_sum)
        if norm == 0.0:
            return (np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int32))
        intensity_sum /= norm
        return uniq, intensity_sum, counts.astype(np.int32, copy=False)

    def _windowed_sum(self, source_bins: np.ndarray, source_vals: np.ndarray,
                      query_positions: np.ndarray, R: int) -> np.ndarray:
        """
        Windowed sum helper: numba-accelerated where available, else vectorized prefix sums.

        Parameters
        ----------
        source_bins : np.ndarray
            Sorted 1D array of integer bin indices (ascending).
        source_vals : np.ndarray
            1D array of values aligned to `source_bins`.
        query_positions : np.ndarray
            1D array of integer bin positions at which to evaluate the windowed sum.
        R : int
            Window radius in bins (inclusive on both sides).

        Returns
        -------
        np.ndarray
            1D array of windowed sums at `query_positions`.
        """
        if source_bins.size == 0 or query_positions.size == 0:
            return np.zeros(query_positions.size, dtype=np.float32)
        if self.use_numba:
            return _windowed_sum_numba(source_bins, source_vals, query_positions, int(R))
        # Vectorized fallback via prefix sums + searchsorted
        c = np.empty(source_vals.size + 1, dtype=np.float32)
        c[0] = 0.0
        c[1:] = np.cumsum(source_vals)
        left = np.searchsorted(source_bins, query_positions - R, side="left")
        right = np.searchsorted(source_bins, query_positions + R, side="right")
        return c[right] - c[left]

    # ---------- Sparse builders for matrix() ----------

    @staticmethod
    def _build_intensity_csr(prepped_list, n_rows: int, offset: int):
        """
        Build CSR matrix (rows=bins, cols=spectra) for intensities.

        Parameters
        ----------
        prepped_list : list of tuples
            Each item is (bins, vals, counts) as returned by `_prep_spectrum`.
        n_rows : int
            Number of bin rows in the global axis.
        offset : int
            Row offset to shift bin indices into [0, n_rows).

        Returns
        -------
        scipy.sparse.csr_array
            CSR with shape (n_rows, n_spectra).
        """
        col_indices = []
        row_indices = []
        data = []
        for j, (bins, vals, _cnts) in enumerate(prepped_list):
            if bins.size == 0:
                continue
            row_indices.append(bins + offset)
            col_indices.append(np.full(bins.size, j, dtype=np.int32))
            data.append(vals.astype(np.float32, copy=False))
        if not row_indices:
            return csr_array((n_rows, len(prepped_list)), dtype=np.float32)
        rows = np.concatenate(row_indices)
        cols = np.concatenate(col_indices)
        dat = np.concatenate(data)
        return csr_array((dat, (rows, cols)), shape=(n_rows, len(prepped_list)), dtype=np.float32)

    @staticmethod
    def _expand_column_blur(rows: np.ndarray, vals: np.ndarray, R: int, n_rows: int):
        """
        Expand one column's nonzeros to ±R neighbors (box blur).

        Parameters
        ----------
        rows : np.ndarray
            Row indices (bins) of nonzero entries for a single column.
        vals : np.ndarray
            Values aligned with `rows`.
        R : int
            Blur radius in bins. If 0, returns inputs unchanged.
        n_rows : int
            Total number of rows (for bounds checking).

        Returns
        -------
        (np.ndarray, np.ndarray)
            (rows_expanded, data_expanded) with out-of-bounds rows removed.
        """
        if rows.size == 0:
            return rows, vals
        # Neighbors offsets
        if R == 0:
            # no expansion
            return rows, vals
        offs = np.arange(-R, R + 1, dtype=np.int32)  # length = 2R+1
        # Broadcast-add and flatten
        neigh = (rows[:, None] + offs[None, :]).ravel()
        data = np.repeat(vals.astype(np.float32, copy=False), offs.size)
        # Clip to valid [0, n_rows)
        mask = (neigh >= 0) & (neigh < n_rows)
        return neigh[mask], data[mask]

    def _build_blurred_csr(self, prepped_list, n_rows: int, offset: int, R: int):
        """
        Build CSR (rows=bins, cols=batch_size) of blurred, normalized query columns.

        Parameters
        ----------
        prepped_list : list of tuples
            Batch of prepped spectra, each as (bins, vals, counts).
        n_rows : int
            Number of rows in the global bin axis.
        offset : int
            Offset applied to bins -> rows.
        R : int
            Blur radius in bins.

        Returns
        -------
        scipy.sparse.csr_array
            CSR matrix (n_rows, batch_size) with duplicated entries summed.
        """
        rows_all = []
        cols_all = []
        data_all = []
        for j, (bins, vals, _cnts) in enumerate(prepped_list):
            if bins.size == 0:
                continue
            base_rows = bins + offset
            rows_exp, data_exp = self._expand_column_blur(base_rows, vals, R, n_rows)
            if rows_exp.size == 0:
                continue
            rows_all.append(rows_exp)
            cols_all.append(np.full(rows_exp.size, j, dtype=np.int32))
            data_all.append(data_exp)

        if not rows_all:
            return csr_array((n_rows, len(prepped_list)), dtype=np.float32)

        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        dat = np.concatenate(data_all)
        # COO -> CSR automatically sums duplicates (needed when blur windows overlap)
        return csr_array((dat, (rows, cols)), shape=(n_rows, len(prepped_list)), dtype=np.float32)
