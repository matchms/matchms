"""SpectraCollection-native Flash similarity implementations.
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import platform
from collections.abc import Sequence
import numpy as np
from numba import njit
from scipy.sparse import coo_array
from tqdm import tqdm
from matchms.scores import Scores
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarity
from .flash_index import FlashIndex
from .default_parameters import (
    DEFAULT_DTYPE,
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_utils import (
    _build_library_index_from_prepared,
    _prepare_collection,
    _PreparedSpectra,
)


logger = logging.getLogger("matchms")


class _BaseFlashSimilarity(BaseSimilarity):
    """Shared base class for SpectraCollection-native Flash similarities."""

    is_commutative = True

    def __init__(
        self,
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        intensity_power: float = DEFAULT_INTENSITY_POWER,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float = DEFAULT_NOISE_CUTOFF,
        normalize_to_half: bool = False,
        merge_within: float = 0,
        identity_precursor_tolerance: float | None = None,
        identity_use_ppm: bool = False,
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        if matching_mode not in ("fragment", "neutral_loss", "hybrid"):
            raise ValueError(
                "matching_mode must be 'fragment', 'neutral_loss', or 'hybrid'"
            )

        self.matching_mode = matching_mode
        self.tolerance = tolerance
        self.use_ppm = use_ppm
        self.intensity_power = intensity_power
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor
        self.noise_cutoff = noise_cutoff
        self.normalize_to_half = normalize_to_half
        self.merge_within = merge_within
        self.identity_precursor_tolerance = identity_precursor_tolerance
        self.identity_use_ppm = identity_use_ppm
        self.dtype = np.dtype(dtype)

    @property
    def _weighing_type(self) -> str:
        raise NotImplementedError

    @property
    def _compute_l2(self) -> bool:
        raise NotImplementedError

    @property
    def _worker(self):
        raise NotImplementedError

    @property
    def _descriptor_name(self) -> str:
        raise NotImplementedError

    @property
    def _search_batch_worker(self):
        raise NotImplementedError

    @staticmethod
    def _as_spectra_collection(spectra):
        """Return input as a SpectraCollection without reconstructing existing collections."""
        from matchms.spectra_collection import SpectraCollection

        if isinstance(spectra, SpectraCollection):
            return spectra
        return SpectraCollection(spectra)

    def _index_config(self) -> dict:
        """Return parameters that materially affect Flash index construction."""
        return {
            "weighing_type": self._weighing_type,
            "compute_l2_norm": bool(self._compute_l2),
            "compute_neutral_loss": self.matching_mode in ("neutral_loss", "hybrid"),
            "intensity_power": float(self.intensity_power),
            "remove_precursor": bool(self.remove_precursor),
            "offset_to_precursor": float(self.offset_to_precursor),
            "noise_cutoff": (
                None if self.noise_cutoff is None else float(self.noise_cutoff)
            ),
            "normalize_to_half": bool(self.normalize_to_half),
            "merge_within": float(self.merge_within),
            "dtype": self.dtype.str,
        }

    @staticmethod
    def _index_metadata(collection) -> dict:
        """Return descriptive source metadata that is not used for compatibility checks."""
        fragments = collection.fragments
        return {
            "n_spectra": int(len(collection)),
            "mz_precision": (
                None
                if getattr(collection, "mz_precision", None) is None
                else float(collection.mz_precision)
            ),
            "mz_rounding": getattr(fragments, "mz_rounding", None),
            "fragment_backend": fragments.__class__.__name__,
        }

    def _validate_index(self, index: FlashIndex) -> None:
        """Validate that a persistent index is compatible with this similarity."""
        if not isinstance(index, FlashIndex):
            raise TypeError(
                "library_index must be a FlashIndex. "
                f"Got {type(index).__name__}."
            )

        expected = self._index_config()
        actual = index.config or {}

        # Neutral-loss-capable indices are supersets and can also serve fragment-only
        # searches. A fragment-only index cannot serve neutral-loss/hybrid searches.
        if (
            expected["compute_neutral_loss"]
            and not actual.get("compute_neutral_loss", False)
        ):
            raise ValueError(
                "The FlashIndex does not contain neutral-loss search arrays, but "
                f"matching_mode={self.matching_mode!r} requires them."
            )

        for key, expected_value in expected.items():
            if key == "compute_neutral_loss":
                continue
            actual_value = actual.get(key, None)
            if actual_value != expected_value:
                raise ValueError(
                    "FlashIndex is incompatible with this similarity configuration: "
                    f"{key}={actual_value!r} in index, expected {expected_value!r}."
                )

        if self._compute_l2 and index.spec_l2 is None:
            raise ValueError("Cosine FlashIndex is missing per-spectrum L2 norms.")
        if (
            self.matching_mode in ("neutral_loss", "hybrid")
            and not index.has_neutral_loss_index
        ):
            raise ValueError(
                "FlashIndex is missing neutral-loss arrays required by the selected "
                "matching mode."
            )

    def build_index(self, spectra) -> FlashIndex:
        """Build a reusable Flash index directly from a SpectraCollection.

        Existing ``SpectraCollection`` inputs stay collection-native throughout:
        preprocessing operates once on the CSR fragment backend and the resulting
        ``_PreparedSpectra`` is passed directly to
        ``_build_library_index_from_prepared``. No ``Spectrum`` objects are
        reconstructed for library construction.
        """
        collection = self._as_spectra_collection(spectra)
        prepared = self._prepare_collection(collection)
        library = self._build_library(prepared)
        return FlashIndex.from_library(
            library,
            config=self._index_config(),
            metadata=self._index_metadata(collection),
        )

    def save_index(self, index: FlashIndex, filename) -> None:
        """Validate and save a reusable Flash index."""
        self._validate_index(index)
        index.save(filename)

    def load_index(self, filename) -> FlashIndex:
        """Load a persistent Flash index and validate it for this similarity."""
        index = FlashIndex.load(filename)
        self._validate_index(index)
        return index

    def _prepare_collection(self, collection) -> _PreparedSpectra:
        """Prepare one complete SpectraCollection for Flash scoring."""
        return _prepare_collection(
            collection,
            intensity_power=self.intensity_power,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=self.normalize_to_half,
            merge_within_da=self.merge_within,
            weighing_type=self._weighing_type,
            compute_l2_norm=self._compute_l2,
            dtype=self.dtype,
        )

    def _prepare_matrix_inputs(self, spectra_1, spectra_2):
        """Convert matrix inputs to collections and preprocess each collection once."""
        collection_1 = self._as_spectra_collection(spectra_1)

        if spectra_2 is None:
            prepared_1 = self._prepare_collection(collection_1)
            return prepared_1, prepared_1, True

        collection_2 = self._as_spectra_collection(spectra_2)
        prepared_1 = self._prepare_collection(collection_1)

        # Reuse preprocessing when the exact same collection is passed explicitly.
        if collection_2 is collection_1:
            prepared_2 = prepared_1
        else:
            prepared_2 = self._prepare_collection(collection_2)

        return prepared_1, prepared_2, False

    def _optimize_matrix_orientation(
        self,
        refs: _PreparedSpectra,
        queries: _PreparedSpectra,
        is_symmetric: bool,
    ) -> tuple[_PreparedSpectra, _PreparedSpectra, bool]:
        """Orient an asymmetric comparison for efficient Flash searching.

        Flash scoring is computationally asymmetric: reference spectra are streamed
        one row at a time, while query spectra are indexed once as the library.
        For commutative similarities it is therefore usually faster to stream the
        smaller collection and build the library from the larger collection.

        If the inputs are swapped internally, the resulting score matrix must be
        transposed before returning it so that the public result still has shape
        ``(len(spectra_1), len(spectra_2))``.

        Parameters
        ----------
        refs
            Prepared first input collection.
        queries
            Prepared second input collection.
        is_symmetric
            Whether this is a self-comparison originating from ``spectra_2=None``.

        Returns
        -------
        refs
            Prepared collection to stream as reference rows.
        queries
            Prepared collection to use as the indexed library.
        transpose_output
            Whether the computed matrix must be transposed before returning.
        """
        if (
            is_symmetric
            or not self.is_commutative
            or refs.n_specs <= queries.n_specs
        ):
            return refs, queries, False

        return queries, refs, True

    def _build_library(self, prepared: _PreparedSpectra):
        return _build_library_index_from_prepared(
            prepared,
            compute_neutral_loss=self.matching_mode in ("neutral_loss", "hybrid"),
            compute_l2_norm=self._compute_l2,
        )

    def _make_worker_cfg(self) -> dict:
        return {
            "tol": float(self.tolerance),
            "use_ppm": bool(self.use_ppm),
            "matching_mode": self.matching_mode,
            "compute_nl": self.matching_mode in ("neutral_loss", "hybrid"),
            "iden_tol": (
                None
                if self.identity_precursor_tolerance is None
                else float(self.identity_precursor_tolerance)
            ),
            "iden_use_ppm": bool(self.identity_use_ppm),
        }

    def _run_row_workers(
        self,
        refs: _PreparedSpectra,
        lib,
        cfg,
        progress_bar: bool,
        n_jobs: int,
        descriptor: str,
    ):
        """Run row workers using integer row ids instead of pickled peak arrays."""
        _set_globals(refs, lib, cfg)
        worker = self._worker
        row_indices = range(refs.n_specs)
        results = []

        if n_jobs in (None, 1, 0):
            for row_idx in tqdm(
                row_indices,
                total=refs.n_specs,
                desc=descriptor + " (matrix)",
                disable=not progress_bar,
            ):
                results.append(worker(row_idx))
            return results

        if platform.system() == "Windows":
            print(
                f"{self.__class__.__name__}.matrix: n_jobs != 1 is not yet "
                "implemented on Windows; falling back to n_jobs=1."
            )
            for row_idx in tqdm(
                row_indices,
                total=refs.n_specs,
                desc=descriptor + " (matrix)",
                disable=not progress_bar,
            ):
                results.append(worker(row_idx))
            return results

        n_cpus = mp.cpu_count()
        if n_jobs < 0:
            n_jobs = max(1, n_cpus + 1 + n_jobs)
        n_jobs = max(1, min(n_jobs, n_cpus))

        start_methods = mp.get_all_start_methods()
        use_fork = "fork" in start_methods and platform.system() != "Windows"

        if not use_fork:
            print(
                f"{self.__class__.__name__}.matrix: parallel execution requires "
                "'fork'; falling back to n_jobs=1."
            )
            for row_idx in tqdm(
                row_indices,
                total=refs.n_specs,
                desc=descriptor + " (matrix)",
                disable=not progress_bar,
            ):
                results.append(worker(row_idx))
            return results

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for result in tqdm(
                pool.imap(worker, row_indices, chunksize=8),
                total=refs.n_specs,
                desc=descriptor + f" (parallel x{n_jobs})",
                disable=not progress_bar,
            ):
                results.append(result)

        return results

    def _run_search_workers(
        self,
        refs: _PreparedSpectra,
        lib: FlashIndex,
        cfg: dict,
        progress_bar: bool,
        n_jobs: int,
        query_batch_size: int,
    ):
        """Score query batches and return only sparse retained hits from workers."""
        _set_globals(refs, lib, cfg)
        worker = self._search_batch_worker
        batches = [
            tuple(range(start, min(start + query_batch_size, refs.n_specs)))
            for start in range(0, refs.n_specs, query_batch_size)
        ]

        results = []
        if not batches:
            return results

        def run_serial(description: str):
            with tqdm(
                total=refs.n_specs,
                desc=description,
                disable=not progress_bar,
            ) as pbar:
                for batch in batches:
                    result = worker(batch)
                    results.append(result)
                    pbar.update(result[0])

        if n_jobs in (None, 1, 0):
            run_serial(self._descriptor_name + " (search)")
            return results

        if platform.system() == "Windows":
            print(
                f"{self.__class__.__name__}.search: n_jobs != 1 is not yet "
                "implemented on Windows; falling back to n_jobs=1."
            )
            run_serial(self._descriptor_name + " (search)")
            return results

        n_cpus = mp.cpu_count()
        if n_jobs < 0:
            n_jobs = max(1, n_cpus + 1 + n_jobs)
        n_jobs = max(1, min(n_jobs, n_cpus))

        start_methods = mp.get_all_start_methods()
        use_fork = "fork" in start_methods and platform.system() != "Windows"
        if not use_fork:
            print(
                f"{self.__class__.__name__}.search: parallel execution requires "
                "'fork'; falling back to n_jobs=1."
            )
            run_serial(self._descriptor_name + " (search)")
            return results

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            with tqdm(
                total=refs.n_specs,
                desc=self._descriptor_name + f" (search parallel x{n_jobs})",
                disable=not progress_bar,
            ) as pbar:
                for result in pool.imap(worker, batches, chunksize=1):
                    results.append(result)
                    pbar.update(result[0])

        return results

    def search(
        self,
        query_spectra,
        library_index: FlashIndex,
        *,
        precursor_tolerance: float | None = None,
        precursor_use_ppm: bool = False,
        min_score: float = 0.0,
        min_matches: int | None = None,
        top_k: int | None = 20,
        query_batch_size: int = 10,
        n_jobs: int = -1,
        progress_bar: bool = True,
    ) -> Scores:
        """Search query spectra against a persistent Flash library index.

        The query side is preprocessed collection-wise. Workers process batches of
        query row ids, reduce each dense internal Flash row immediately by
        ``min_score``/``min_matches``/``top_k``, and return only compact sparse
        hits to the parent process. Consequently no dense
        ``n_queries x n_library`` result matrix is materialized and dense
        library-length rows are not transferred between processes.
        """
        self._validate_index(library_index)

        if precursor_tolerance is not None and precursor_tolerance < 0:
            raise ValueError("precursor_tolerance must be None or >= 0.")
        if min_score < 0:
            raise ValueError("min_score must be >= 0.")
        if min_matches is not None and min_matches < 0:
            raise ValueError("min_matches must be None or >= 0.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be None or a positive integer.")
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be a positive integer.")
        if self._weighing_type == "entropy" and min_matches is not None:
            raise ValueError("min_matches is only available for CosineFlash search.")

        collection = self._as_spectra_collection(query_spectra)
        refs = self._prepare_collection(collection)

        cfg = self._make_worker_cfg()
        if precursor_tolerance is not None:
            cfg["iden_tol"] = float(precursor_tolerance)
            cfg["iden_use_ppm"] = bool(precursor_use_ppm)
        cfg["search_min_score"] = float(min_score)
        cfg["search_min_matches"] = (
            None if min_matches is None else int(min_matches)
        )
        cfg["search_top_k"] = None if top_k is None else int(top_k)
        cfg["search_require_precursor"] = cfg["iden_tol"] is not None

        batch_results = self._run_search_workers(
            refs=refs,
            lib=library_index,
            cfg=cfg,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            query_batch_size=int(query_batch_size),
        )

        shape = (refs.n_specs, library_index.n_specs)
        if self._weighing_type == "entropy":
            rows, cols, scores = _combine_entropy_search_batches(batch_results)
            return Scores(
                {
                    "score": coo_array(
                        (scores.astype(self.dtype, copy=False), (rows, cols)),
                        shape=shape,
                    )
                }
            )

        rows, cols, scores, matches = _combine_cosine_search_batches(batch_results)
        return Scores(
            {
                "score": coo_array(
                    (scores.astype(self.dtype, copy=False), (rows, cols)),
                    shape=shape,
                ),
                "matches": coo_array(
                    (matches.astype(np.int32, copy=False), (rows, cols)),
                    shape=shape,
                ),
            }
        )


class FlashEntropy(_BaseFlashSimilarity):
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
    matching_mode:
        Strategy used to match peaks between spectra.

        - ``"fragment"``:
        Match fragment m/z values directly. Each peak can be matched at most once.

        - ``"neutral_loss"``:
        Match neutral losses (precursor_mz - fragment_mz) only. Each peak can be
        matched at most once. If either spectrum lacks precursor m/z metadata,
        the score for that pair is zero.

        - ``"hybrid"``:
        First perform one-to-one fragment matching. Peaks consumed by fragment
        matches are excluded from subsequent neutral-loss matching. Remaining
        peaks are then matched one-to-one by neutral loss. If either spectrum
        lacks precursor m/z metadata, hybrid scoring falls back to fragment-only
        scoring for that pair.

    Default is ``"fragment"``.
    tolerance:
        Matching tolerance in Da or ppm (use_ppm=True). Default is 0.01.
    use_ppm:
        If True, interpret `tolerance` as parts-per-million. Default is False.
    remove_precursor:
        If True, remove precursor peak and peaks within offset_to_precursor.
        Default is True.
    offset_to_precursor:
        Offset used when ``remove_precursor=True``. This will only keep 
        mz values <= precursor_mz + offset_to_precursor. Default is -1.6 Da.
        (as suggested by Li & Fiehn(2023)).
    noise_cutoff:
        If > 0, remove peaks with intensities below this fraction of the maximum intensity.
        Default is 0.01 (1%).
    normalize_to_half:
        If True, normalize intensities such that the sum of intensities is 0.5.
        Default is True.
    merge_within:
        If > 0, merge peaks within this distance (in Da) to a single peak.
        Default is 0.
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
    score_fields = ("score",)

    def __init__(
        self,
        *args,
        normalize_to_half: bool = True,
        dtype: np.dtype = DEFAULT_DTYPE,
        **kwargs,
    ):
        super().__init__(
            *args,
            normalize_to_half=normalize_to_half,
            dtype=dtype,
            **kwargs,
        )
        self.score_datatype = self.dtype

    @property
    def _weighing_type(self) -> str:
        return "entropy"

    @property
    def _compute_l2(self) -> bool:
        return False

    @property
    def _worker(self):
        return _row_task_entropy

    @property
    def _search_batch_worker(self):
        return _search_batch_task_entropy

    @property
    def _descriptor_name(self) -> str:
        return f"Flash entropy ({self.matching_mode})"

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """
        Compute Flash Entropy for a single (reference, query) pair.
        Uses the same preprocessing and scoring logic as the matrix path, but builds a tiny
        1-spectrum library from the query.
        
        Careful: This is not the fast intended use; better .matrix() instead.
        """
        logger.warning("This is not the fast intended use; better use .matrix() instead.")
        scores = self.matrix(
            [spectrum_1],
            [spectrum_2],
            score_fields=("score",),
            progress_bar=False,
            n_jobs=0,
        )
        return np.asarray(scores.to_array("score")[0, 0], dtype=self.dtype)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ) -> Scores:
        """
        Calculate matrix of Flash Entropy scores.

        Parameters
        ----------
        spectra_1
            First collection of input spectra.
        spectra_2
            Second collection of input spectra. If None, compare `spectra_1`
            against itself.
        score_fields
            Requested score fields. Only ``("score",)`` is supported.
        progress_bar
            When True, show a progress bar.
        n_jobs
            Number of parallel jobs to run.
            Default is -1, which means that all available CPUs minus one will be used.

        Returns
        -------
        Scores
            Dense score matrix as a ``Scores`` object.
        """
        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError(
                "FlashEntropy.matrix() supports only score_fields=('score',)."
            )

        refs, queries, is_symmetric = self._prepare_matrix_inputs(
            spectra_1,
            spectra_2,
        )
        if is_symmetric and refs.n_specs != queries.n_specs:
            raise ValueError(
                "Self-comparison requires same number of rows and columns."
            )

        refs, queries, transpose_output = self._optimize_matrix_orientation(
            refs,
            queries,
            is_symmetric,
        )

        lib = self._build_library(queries)
        results = self._run_row_workers(
            refs=refs,
            lib=lib,
            cfg=self._make_worker_cfg(),
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            descriptor=self._descriptor_name,
        )

        out_score = np.zeros(
            (refs.n_specs, queries.n_specs),
            dtype=self.dtype,
        )
        for row_idx, row_score in results:
            out_score[row_idx, :] = row_score

        if transpose_output:
            out_score = out_score.T

        return Scores(
            {
                "score": out_score.astype(
                    self.score_datatype,
                    copy=False,
                )
            }
        )


class CosineFlash(_BaseFlashSimilarity):
    """
    Flash Cosine similarity following the original Flash Entropy (Li & Fiehn, 2023)
    with a fast .matrix() that builds a library-wide index over 'queries' and streams 
    all 'references' through it. This corresponds to the "CosineGreedy" scoring logic
    but with the same fast Flash path as Flash Entropy.

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
    matching_mode:
        Matching mode: 'fragment', 'neutral_loss', or 'hybrid' (default is 'fragment').
    tolerance:
        Matching tolerance in Da or ppm (use_ppm=True). Default is 0.01.
    use_ppm:
        If True, interpret `tolerance` as parts-per-million. Default is False.
    intensity_power:
        The power to raise intensity to in the cosine function. The default is 1 (no weighting).
    remove_precursor:
        If True, remove precursor peak and peaks within offset_to_precursor.
        Default is True.
    offset_to_precursor:
        If remove_precursor is True, remove peaks within this window around the precursor
        m/z. Default is 1.6 Da (as suggested by Li & Fiehn(2023)).
    noise_cutoff:
        If > 0, remove peaks with intensities below this fraction of the maximum intensity.
        Default is 0.01 (1%).
    normalize_to_half:
        If True, normalize intensities such that the sum of intensities is 0.5.
        Default is False.
    merge_within:
        If > 0, merge peaks within this distance (in Da) to a single peak.
        Default is 0.
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

    score_fields = ("score", "matches")

    def __init__(self, *args, dtype: np.dtype = DEFAULT_DTYPE, **kwargs):
        super().__init__(*args, dtype=dtype, **kwargs)
        self.score_datatype = np.dtype(
            [("score", self.dtype), ("matches", np.int32)]
        )

    @property
    def _weighing_type(self) -> str:
        return "cosine"

    @property
    def _compute_l2(self) -> bool:
        return True

    @property
    def _worker(self):
        return _row_task_cosine

    @property
    def _search_batch_worker(self):
        return _search_batch_task_cosine

    @property
    def _descriptor_name(self) -> str:
        return f"Flash cosine ({self.matching_mode})"

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """Compute one score via the same collection-native matrix path."""
        logger.warning("CosineFlash.pair() is not the fast intended use; use .matrix().")
        scores = self.matrix(
            [spectrum_1],
            [spectrum_2],
            score_fields=("score", "matches"),
            progress_bar=False,
            n_jobs=0,
        )
        return np.asarray(
            (
                scores.to_array("score")[0, 0],
                scores.to_array("matches")[0, 0],
            ),
            dtype=self.score_datatype,
        )

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ) -> Scores:
        """
        Calculate matrix of Flash Cosine scores.

        Parameters
        ----------
        spectra_1
            First collection of input spectra.
        spectra_2
            Second collection of input spectra. If None, compare `spectra_1`
            against itself.
        score_fields
            Requested score fields. Only ``("score",)`` is supported.
        progress_bar
            When True, show a progress bar.
        n_jobs
            Number of parallel jobs to run.
            Default is -1, which means that all available CPUs minus one will be used.

        Returns
        -------
        Scores
            Dense score matrix as a ``Scores`` object.
        """
        selected_fields = self._resolve_score_fields(score_fields)

        refs, queries, is_symmetric = self._prepare_matrix_inputs(
            spectra_1,
            spectra_2,
        )
        if is_symmetric and refs.n_specs != queries.n_specs:
            raise ValueError(
                "Self-comparison requires same number of rows and columns."
            )

        refs, queries, transpose_output = self._optimize_matrix_orientation(
            refs,
            queries,
            is_symmetric,
        )

        lib = self._build_library(queries)
        results = self._run_row_workers(
            refs=refs,
            lib=lib,
            cfg=self._make_worker_cfg(),
            progress_bar=progress_bar,
            n_jobs=n_jobs,
            descriptor=self._descriptor_name,
        )

        out_score = np.zeros(
            (refs.n_specs, queries.n_specs),
            dtype=self.dtype,
        )
        out_matches = np.zeros(
            (refs.n_specs, queries.n_specs),
            dtype=np.int32,
        )

        for row_idx, row_score, row_matches in results:
            out_score[row_idx, :] = row_score
            out_matches[row_idx, :] = row_matches

        if transpose_output:
            out_score = out_score.T
            out_matches = out_matches.T

        result = {}

        if "score" in selected_fields:
            result["score"] = out_score.astype(
                self.dtype,
                copy=False,
            )

        if "matches" in selected_fields:
            result["matches"] = out_matches

        return Scores(result)


# ===================== worker plumbing =====================

_G_REFS = None
_G_LIB = None
_G_CFG = None


def _set_globals(refs_obj, lib_obj, cfg):
    """Install packed references, library index, and worker configuration."""
    global _G_REFS, _G_LIB, _G_CFG
    _G_REFS = refs_obj
    _G_LIB = lib_obj
    _G_CFG = cfg

@njit(cache=True, nogil=True)
def _search_spec_in_fragment_window(lib_mz: np.ndarray, mz: float, tol: float, use_ppm: bool):
    """
    Find the range of indices in lib_mz that fall within the tolerance window of mz.
    Returns (a, b) such that lib_mz[a:b] are the candidates.
    """
    mz_tolerance = _search_window_halfwidth_nb(mz, tol, use_ppm)
    lo_mz = mz - mz_tolerance
    hi_mz = mz + mz_tolerance

    a = np.searchsorted(lib_mz, lo_mz, side="left")
    b = np.searchsorted(lib_mz, hi_mz, side="right")
    return a, b


@njit(cache=True, nogil=True)
def _search_window_halfwidth_nb(m: float, tol: float, use_ppm: bool) -> float:
    """
    Compute half-width of a symmetric search window around mass ``m``.

    If ``use_ppm`` is False: returns ``tol`` (Da half-width).
    If ``use_ppm`` is True: uses symmetric-ppm definition:
        |m2 - m1| <= tol[ppm] * 1e-6 * 0.5 * (m1 + m2)
    which corresponds to a half-window of approximately (tol * 1e-6 * mass),
    corrected for symmetry.

    Notes
    -----
    This helper is only for search-window bounds. Candidate matches are still
    filtered by exact checks in :func:`_within_tol_nb`.
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
def _within_tol_nb(m1: float, m2: float, tol: float, use_ppm: bool) -> bool:
    """
    Numba-fied symmetric tolerance check.

    Symmetric ppm definition:
        |m2 - m1| <= tol[ppm] * 1e-6 * 0.5 * (m1 + m2)
    Otherwise absolute Da tolerance:
        |m2 - m1| <= tol

    This is the exact tolerance predicate used to trim search windows produced
    by :func:`_search_window_halfwidth_nb`.
    """
    if use_ppm:
        return abs(m2 - m1) <= (tol * 1e-6) * 0.5 * (m1 + m2)
    return abs(m2 - m1) <= tol


_ENTROPY_MODE_FRAGMENT = 0
_ENTROPY_MODE_NEUTRAL_LOSS = 1
_ENTROPY_MODE_HYBRID = 2


@njit(cache=True, nogil=True)
def _gather_entropy_candidate_cols_numba(
    ref_mz: np.ndarray,
    ref_intensity: np.ndarray,
    ref_pmz: float,
    has_ref_pmz: bool,
    lib_mz: np.ndarray,
    lib_spec_index: np.ndarray,
    lib_nl_mz: np.ndarray,
    lib_nl_spec_index: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    n_cols: int,
    matching_mode: int,
) -> np.ndarray:
    """Collect library spectra with at least one possible entropy match.

    Global fragment and neutral-loss indices are only used for candidate
    discovery. Exact one-to-one matching is performed later spectrum-by-spectrum.
    """
    if n_cols == 0:
        return np.empty(0, dtype=np.int64)

    do_fragment = matching_mode in (
        _ENTROPY_MODE_FRAGMENT,
        _ENTROPY_MODE_HYBRID,
    )
    do_neutral_loss = matching_mode in (
        _ENTROPY_MODE_NEUTRAL_LOSS,
        _ENTROPY_MODE_HYBRID,
    )

    seen = np.zeros(n_cols, dtype=np.uint8)
    candidate_cols = np.empty(n_cols, dtype=np.int64)
    n_candidates = 0

    # Fragment candidates.
    if do_fragment:
        for i in range(ref_mz.shape[0]):
            if ref_intensity[i] <= 0.0:
                continue

            mz = float(ref_mz[i])
            a, b = _search_spec_in_fragment_window(
                lib_mz,
                mz,
                tolerance,
                use_ppm,
            )

            for j in range(a, b):
                if not _within_tol_nb(
                    mz,
                    float(lib_mz[j]),
                    tolerance,
                    use_ppm,
                ):
                    continue

                col = int(lib_spec_index[j])
                if seen[col] == 0:
                    seen[col] = 1
                    candidate_cols[n_candidates] = col
                    n_candidates += 1

    # Neutral-loss candidates.
    if (
        do_neutral_loss
        and has_ref_pmz
        and lib_nl_mz.size > 0
    ):
        for i in range(ref_mz.shape[0]):
            if ref_intensity[i] <= 0.0:
                continue

            loss = ref_pmz - float(ref_mz[i])
            a, b = _search_spec_in_fragment_window(
                lib_nl_mz,
                loss,
                tolerance,
                use_ppm,
            )

            for j in range(a, b):
                if not _within_tol_nb(
                    loss,
                    float(lib_nl_mz[j]),
                    tolerance,
                    use_ppm,
                ):
                    continue

                col = int(lib_nl_spec_index[j])
                if seen[col] == 0:
                    seen[col] = 1
                    candidate_cols[n_candidates] = col
                    n_candidates += 1

    return candidate_cols[:n_candidates]


@njit(cache=True, nogil=True)
def _entropy_increment_nb(
    intensity_1: float,
    intensity_2: float,
) -> float:
    """Return entropy-similarity contribution of one matched peak pair."""
    return (
        _xlog2_scalar_nb(intensity_1 + intensity_2)
        - _xlog2_scalar_nb(intensity_1)
        - _xlog2_scalar_nb(intensity_2)
    )


@njit(cache=True, nogil=True)
def _entropy_fragment_pair_numba(
    ref_mz: np.ndarray,
    ref_intensity: np.ndarray,
    lib_mz: np.ndarray,
    lib_intensity: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    used_ref: np.ndarray,
    used_lib: np.ndarray,
    track_used: bool,
) -> float:
    """Score one-to-one fragment matches in ascending m/z order."""
    i = 0
    j = 0
    score = 0.0

    while i < ref_mz.shape[0] and j < lib_mz.shape[0]:
        if track_used and used_ref[i] != 0:
            i += 1
            continue

        if track_used and used_lib[j] != 0:
            j += 1
            continue

        intensity_ref = float(ref_intensity[i])
        intensity_lib = float(lib_intensity[j])

        if intensity_ref <= 0.0:
            i += 1
            continue

        if intensity_lib <= 0.0:
            j += 1
            continue

        mz_ref = float(ref_mz[i])
        mz_lib = float(lib_mz[j])

        if _within_tol_nb(
            mz_ref,
            mz_lib,
            tolerance,
            use_ppm,
        ):
            score += _entropy_increment_nb(
                intensity_ref,
                intensity_lib,
            )

            if track_used:
                used_ref[i] = 1
                used_lib[j] = 1

            i += 1
            j += 1

        elif mz_ref < mz_lib:
            i += 1
        else:
            j += 1

    return score


@njit(cache=True, nogil=True)
def _entropy_neutral_loss_pair_numba(
    ref_mz: np.ndarray,
    ref_intensity: np.ndarray,
    ref_pmz: float,
    lib_mz: np.ndarray,
    lib_intensity: np.ndarray,
    lib_pmz: float,
    tolerance: float,
    use_ppm: bool,
    used_ref: np.ndarray,
    used_lib: np.ndarray,
    track_used: bool,
) -> float:
    """Score one-to-one neutral-loss matches in ascending loss order."""
    i = ref_mz.shape[0] - 1
    j = lib_mz.shape[0] - 1
    score = 0.0

    while i >= 0 and j >= 0:
        if track_used and used_ref[i] != 0:
            i -= 1
            continue

        if track_used and used_lib[j] != 0:
            j -= 1
            continue

        intensity_ref = float(ref_intensity[i])
        intensity_lib = float(lib_intensity[j])

        if intensity_ref <= 0.0:
            i -= 1
            continue

        if intensity_lib <= 0.0:
            j -= 1
            continue

        loss_ref = ref_pmz - float(ref_mz[i])
        loss_lib = lib_pmz - float(lib_mz[j])

        if _within_tol_nb(
            loss_ref,
            loss_lib,
            tolerance,
            use_ppm,
        ):
            score += _entropy_increment_nb(
                intensity_ref,
                intensity_lib,
            )

            if track_used:
                used_ref[i] = 1
                used_lib[j] = 1

            i -= 1
            j -= 1

        elif loss_ref < loss_lib:
            # Moving backwards in fragment m/z increases neutral-loss m/z.
            i -= 1
        else:
            j -= 1

    return score


@njit(cache=True, nogil=True)
def _score_entropy_candidate_cols_numba(
    scores: np.ndarray,
    ref_mz: np.ndarray,
    ref_intensity: np.ndarray,
    ref_pmz: float,
    has_ref_pmz: bool,
    candidate_cols: np.ndarray,
    spec_offsets: np.ndarray,
    spec_mz: np.ndarray,
    spec_intensity: np.ndarray,
    lib_precursor_mz: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    matching_mode: int,
) -> None:
    """Calculate exact one-to-one entropy scores for candidate spectra."""
    empty_used = np.empty(0, dtype=np.uint8)

    for candidate_idx in range(candidate_cols.shape[0]):
        col = int(candidate_cols[candidate_idx])

        start = int(spec_offsets[col])
        end = int(spec_offsets[col + 1])

        if start >= end:
            continue

        lib_mz = spec_mz[start:end]
        lib_intensity = spec_intensity[start:end]
        lib_pmz = float(lib_precursor_mz[col])
        has_lib_pmz = not np.isnan(lib_pmz)

        if matching_mode == _ENTROPY_MODE_FRAGMENT:
            scores[col] = _entropy_fragment_pair_numba(
                ref_mz,
                ref_intensity,
                lib_mz,
                lib_intensity,
                tolerance,
                use_ppm,
                empty_used,
                empty_used,
                False,
            )
            continue

        if matching_mode == _ENTROPY_MODE_NEUTRAL_LOSS:
            if not has_ref_pmz or not has_lib_pmz:
                continue

            scores[col] = _entropy_neutral_loss_pair_numba(
                ref_mz,
                ref_intensity,
                ref_pmz,
                lib_mz,
                lib_intensity,
                lib_pmz,
                tolerance,
                use_ppm,
                empty_used,
                empty_used,
                False,
            )
            continue

        # Hybrid:
        # fragment matches have priority. Only peaks not consumed by fragment
        # matching may subsequently participate in neutral-loss matching.
        used_ref = np.zeros(ref_mz.shape[0], dtype=np.uint8)
        used_lib = np.zeros(lib_mz.shape[0], dtype=np.uint8)

        score = _entropy_fragment_pair_numba(
            ref_mz,
            ref_intensity,
            lib_mz,
            lib_intensity,
            tolerance,
            use_ppm,
            used_ref,
            used_lib,
            True,
        )

        if has_ref_pmz and has_lib_pmz:
            score += _entropy_neutral_loss_pair_numba(
                ref_mz,
                ref_intensity,
                ref_pmz,
                lib_mz,
                lib_intensity,
                lib_pmz,
                tolerance,
                use_ppm,
                used_ref,
                used_lib,
                True,
            )

        scores[col] = score




def _row_task_entropy(row_idx):
    """Compute one FlashEntropy row from the shared packed references."""
    row_idx = int(row_idx)
    refs = _G_REFS
    lib = _G_LIB
    cfg = _G_CFG
    scores = np.zeros(lib.n_specs, dtype=lib.dtype)

    start = int(refs.spec_offsets[row_idx])
    end = int(refs.spec_offsets[row_idx + 1])
    if start >= end:
        return row_idx, scores

    ref_mz = refs.spec_mz[start:end]
    ref_int = refs.spec_int[start:end]
    ref_pmz_value = float(refs.precursor_mz[row_idx])
    has_ref_pmz = np.isfinite(ref_pmz_value)

    matching_mode = cfg["matching_mode"]
    if matching_mode == "fragment":
        matching_mode_code = _ENTROPY_MODE_FRAGMENT
    elif matching_mode == "neutral_loss":
        matching_mode_code = _ENTROPY_MODE_NEUTRAL_LOSS
    else:
        matching_mode_code = _ENTROPY_MODE_HYBRID

    if matching_mode_code == _ENTROPY_MODE_FRAGMENT:
        lib_nl_mz = np.empty(0, dtype=lib.dtype)
        lib_nl_spec_idx = np.empty(0, dtype=np.int32)
    else:
        lib_nl_mz = lib.nl_mz if lib.nl_mz is not None else np.empty(0, dtype=lib.dtype)
        lib_nl_spec_idx = (
            lib.nl_spec_idx
            if lib.nl_spec_idx is not None
            else np.empty(0, dtype=np.int32)
        )

    candidate_cols = _gather_entropy_candidate_cols_numba(
        ref_mz,
        ref_int,
        ref_pmz_value,
        has_ref_pmz,
        lib.peaks_mz,
        lib.peaks_spec_idx,
        lib_nl_mz,
        lib_nl_spec_idx,
        float(cfg["tol"]),
        bool(cfg["use_ppm"]),
        lib.n_specs,
        matching_mode_code,
    )

    _score_entropy_candidate_cols_numba(
        scores,
        ref_mz,
        ref_int,
        ref_pmz_value,
        has_ref_pmz,
        candidate_cols,
        lib.spec_offsets,
        lib.spec_mz,
        lib.spec_int,
        lib.precursor_mz,
        float(cfg["tol"]),
        bool(cfg["use_ppm"]),
        matching_mode_code,
    )

    if cfg["iden_tol"] is not None and has_ref_pmz:
        if cfg["iden_use_ppm"]:
            allow = np.abs(lib.precursor_mz - ref_pmz_value) <= (
                cfg["iden_tol"]
                * 1e-6
                * 0.5
                * (lib.precursor_mz + ref_pmz_value)
            )
        else:
            allow = np.abs(lib.precursor_mz - ref_pmz_value) <= cfg["iden_tol"]

        allow &= np.isfinite(lib.precursor_mz)
        scores[~allow] = 0.0

    return row_idx, scores

@njit(cache=True, nogil=True)
def _precursor_shift_is_effectively_zero_nb(
    ref_pmz: float,
    lib_pmz: float,
    tol: float,
    use_ppm: bool,
) -> bool:
    if np.isnan(ref_pmz) or np.isnan(lib_pmz):
        return False
    return _within_tol_nb(ref_pmz, lib_pmz, tol, use_ppm)


@njit(cache=True, nogil=True)
def _count_candidates_per_col_nb(
        query_mz: np.ndarray,
        query_int: np.ndarray,
        has_pmz: bool,
        query_pmz: float,
        peaks_mz: np.ndarray, peaks_spec_idx: np.ndarray,
        nl_mz: np.ndarray, nl_spec_idx: np.ndarray,
        nl_prod_idx: np.ndarray,
        lib_precursor_mz: np.ndarray,
        tol: float, use_ppm: bool,
        do_frag: bool, do_nl: bool,
        n_cols: int
        ) -> np.ndarray:
    """
    Count (fragment + neutral loss) candidate pairs per column (library spectrum).

    Returns
    -------
    counts_by_col : int32[n_cols]
        Number of candidate (i,j) pairs that belong to each library spectrum (column).
    """
    counts = np.zeros(n_cols, dtype=np.int32)

    # Fragment matches
    if do_frag:
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            mz = float(query_mz[i])
            a, b = _search_spec_in_fragment_window(peaks_mz, mz, tol, use_ppm)
            for j in range(a, b):
                # exact symmetric check to trim the searchsorted window
                if _within_tol_nb(mz, float(peaks_mz[j]), tol, use_ppm):
                    col = int(peaks_spec_idx[j])
                    counts[col] += 1

    # Neutral-loss matches
    if do_nl and has_pmz and (nl_mz.size > 0):
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            loss = query_pmz - float(query_mz[i])
            a, b = _search_spec_in_fragment_window(nl_mz, loss, tol, use_ppm)

            for k in range(a, b):
                if not _within_tol_nb(loss, float(nl_mz[k]), tol, use_ppm):
                    continue

                col = int(nl_spec_idx[k])

                # Hybrid ModifiedCosine compatibility:
                # if precursor masses are within tolerance, the shifted/NL branch
                # would duplicate fragment matching and should be ignored.
                if do_frag:
                    lib_pmz = float(lib_precursor_mz[col])
                    if _precursor_shift_is_effectively_zero_nb(query_pmz, lib_pmz, tol, use_ppm):
                        continue

                counts[col] += 1

    return counts


@njit(cache=True, nogil=True)
def _fill_candidates_per_col_nb(
        query_mz: np.ndarray,
        query_int: np.ndarray,
        has_pmz: bool, qpmz: float,
        peaks_mz: np.ndarray, peaks_int: np.ndarray, peaks_spec_idx: np.ndarray,
        nl_mz: np.ndarray, nl_spec_idx: np.ndarray, nl_prod_idx: np.ndarray,
        lib_precursor_mz: np.ndarray,
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
    score = np.empty(n_total, dtype=np.float64)
    is_frag = np.empty(n_total, dtype=np.uint8)

    pos = col_offsets[:-1].copy()

    if do_frag:
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            mz = float(query_mz[i])
            a, b = _search_spec_in_fragment_window(peaks_mz, mz, tol, use_ppm)
            for j in range(a, b):
                mj = float(peaks_mz[j])
                if not _within_tol_nb(mz, mj, tol, use_ppm):
                    continue

                spec_idx = int(peaks_spec_idx[j])
                col_idx = int(pos[spec_idx])
                ref_idx[col_idx] = i
                lib_idx[col_idx] = j
                score[col_idx] = Iq * float(peaks_int[j])
                is_frag[col_idx] = 1
                pos[spec_idx] = col_idx + 1

    if do_nl and has_pmz and (nl_mz.size > 0):
        for i in range(query_mz.shape[0]):
            Iq = float(query_int[i])
            if Iq <= 0.0:
                continue

            loss = qpmz - float(query_mz[i])
            a, b = _search_spec_in_fragment_window(nl_mz, loss, tol, use_ppm)

            for k in range(a, b):
                mk = float(nl_mz[k])
                if not _within_tol_nb(loss, mk, tol, use_ppm):
                    continue

                spec_idx = int(nl_spec_idx[k])

                # Same skip as in the count pass.
                if do_frag:
                    lib_pmz = float(lib_precursor_mz[spec_idx])
                    if _precursor_shift_is_effectively_zero_nb(qpmz, lib_pmz, tol, use_ppm):
                        continue

                j = int(nl_prod_idx[k])
                col_idx = int(pos[spec_idx])
                ref_idx[col_idx] = i
                lib_idx[col_idx] = j
                score[col_idx] = Iq * float(peaks_int[j])
                is_frag[col_idx] = 0
                pos[spec_idx] = col_idx + 1

    return ref_idx, lib_idx, score, is_frag


@njit(cache=True, nogil=True)
def _greedy_scores_and_matches_all_cols_nb(
    n_cols: int,
    n_q: int,
    col_offsets: np.ndarray,
    ref_idx: np.ndarray,
    lib_idx: np.ndarray,
    score: np.ndarray,
    is_frag: np.ndarray,
    lib_spec_l2: np.ndarray,
    q_l2: float,
):
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
    out_score = np.zeros(n_cols, dtype=np.float64)
    out_matches = np.zeros(n_cols, dtype=np.int32)
    used_q = np.empty(n_q, dtype=np.uint8)

    eps = 1e-12

    for col in range(n_cols):
        start = int(col_offsets[col])
        end = int(col_offsets[col + 1])
        size = end - start
        if size <= 0:
            continue

        s_ref = ref_idx[start:end]
        s_lib = lib_idx[start:end]
        s_score = score[start:end]
        s_frag = is_frag[start:end]

        key = np.empty(size, dtype=np.float64)
        for t in range(size):
            key[t] = s_score[t] + (eps if s_frag[t] == 1 else 0.0)
        order = np.argsort(-key)

        for qi in range(n_q):
            used_q[qi] = 0

        used_lib_j = np.empty(size, dtype=np.int32)
        used_lib_n = 0

        dot = 0.0
        n_match = 0

        for r in range(size):
            idx = int(order[r])
            i = int(s_ref[idx])
            j = int(s_lib[idx])

            if used_q[i] == 1:
                continue

            seen = False
            for u in range(used_lib_n):
                if used_lib_j[u] == j:
                    seen = True
                    break
            if seen:
                continue

            used_q[i] = 1
            used_lib_j[used_lib_n] = j
            used_lib_n += 1
            dot += float(s_score[idx])
            n_match += 1

        denom = q_l2 * float(lib_spec_l2[col])
        if denom > 0.0 and dot > 0.0:
            out_score[col] = dot / denom
        out_matches[col] = n_match

    return out_score, out_matches




def _row_task_cosine(row_idx):
    """Compute one CosineFlash row from the shared packed references."""
    row_idx = int(row_idx)
    refs = _G_REFS
    lib = _G_LIB
    cfg = _G_CFG

    start = int(refs.spec_offsets[row_idx])
    end = int(refs.spec_offsets[row_idx + 1])
    if start >= end:
        return (
            row_idx,
            np.zeros(lib.n_specs, dtype=lib.dtype),
            np.zeros(lib.n_specs, dtype=np.int32),
        )

    ref_mz = refs.spec_mz[start:end]
    ref_int = refs.spec_int[start:end]

    if refs.spec_l2 is not None:
        q_l2 = float(refs.spec_l2[row_idx])
    else:
        q_l2 = float(np.sqrt(np.sum(ref_int * ref_int, dtype=np.float64)))

    if q_l2 == 0.0:
        return (
            row_idx,
            np.zeros(lib.n_specs, dtype=lib.dtype),
            np.zeros(lib.n_specs, dtype=np.int32),
        )

    match_mode = cfg["matching_mode"]
    do_frag = match_mode in ("fragment", "hybrid")
    do_nl = match_mode in ("neutral_loss", "hybrid")

    ref_pmz = float(refs.precursor_mz[row_idx])
    has_pmz = np.isfinite(ref_pmz)
    if not has_pmz:
        ref_pmz = 0.0

    tol = float(cfg["tol"])
    use_ppm = bool(cfg["use_ppm"])
    n_cols = int(lib.n_specs)

    empty_mz = np.empty(0, dtype=lib.dtype)
    empty_idx = np.empty(0, dtype=np.int32)
    empty_prod = np.empty(0, dtype=np.int64)

    lib_nl_mz = (
        lib.nl_mz
        if cfg["compute_nl"] and lib.nl_mz is not None
        else empty_mz
    )
    lib_nl_spec_idx = (
        lib.nl_spec_idx
        if cfg["compute_nl"] and lib.nl_spec_idx is not None
        else empty_idx
    )
    lib_nl_product_idx = (
        lib.nl_product_idx
        if cfg["compute_nl"] and lib.nl_product_idx is not None
        else empty_prod
    )

    counts_by_col = _count_candidates_per_col_nb(
        ref_mz,
        ref_int,
        has_pmz,
        ref_pmz,
        lib.peaks_mz,
        lib.peaks_spec_idx.astype(np.int32, copy=False),
        lib_nl_mz,
        lib_nl_spec_idx.astype(np.int32, copy=False),
        lib_nl_product_idx,
        lib.precursor_mz.astype(np.float64, copy=False),
        tol,
        use_ppm,
        do_frag,
        do_nl and cfg["compute_nl"],
        n_cols,
    )

    col_offsets = np.empty(n_cols + 1, dtype=np.int64)
    col_offsets[0] = 0
    for col in range(n_cols):
        col_offsets[col + 1] = col_offsets[col] + counts_by_col[col]

    n_total = int(col_offsets[-1])
    if n_total == 0:
        out_score = np.zeros(n_cols, dtype=lib.dtype)
        out_matches = np.zeros(n_cols, dtype=np.int32)
    else:
        ref_idx, lib_idx, score, is_frag = _fill_candidates_per_col_nb(
            ref_mz,
            ref_int,
            has_pmz,
            ref_pmz,
            lib.peaks_mz,
            lib.peaks_int,
            lib.peaks_spec_idx.astype(np.int32, copy=False),
            lib_nl_mz,
            lib_nl_spec_idx.astype(np.int32, copy=False),
            lib_nl_product_idx,
            lib.precursor_mz.astype(np.float64, copy=False),
            tol,
            use_ppm,
            do_frag,
            do_nl and cfg["compute_nl"],
            col_offsets,
            counts_by_col,
        )

        out_score64, out_matches = _greedy_scores_and_matches_all_cols_nb(
            n_cols,
            ref_mz.shape[0],
            col_offsets,
            ref_idx,
            lib_idx,
            score,
            is_frag,
            lib.spec_l2.astype(np.float64, copy=False),
            q_l2,
        )
        out_score = out_score64.astype(lib.dtype, copy=False)

    iden_tol = cfg["iden_tol"]
    if iden_tol is not None and has_pmz:
        if cfg["iden_use_ppm"]:
            allow = np.abs(lib.precursor_mz - ref_pmz) <= (
                iden_tol * 1e-6 * 0.5 * (lib.precursor_mz + ref_pmz)
            )
        else:
            allow = np.abs(lib.precursor_mz - ref_pmz) <= iden_tol
        allow &= np.isfinite(lib.precursor_mz)
        out_score[~allow] = 0.0
        out_matches[~allow] = 0

    return row_idx, out_score, out_matches


# ===================== sparse search reduction =====================

def _select_search_hit_columns(scores, matches, cfg):
    """Return retained library columns sorted by descending score."""
    keep = scores > 0.0
    keep &= scores >= float(cfg["search_min_score"])

    min_matches = cfg.get("search_min_matches")
    if min_matches is not None:
        keep &= matches >= int(min_matches)

    cols = np.flatnonzero(keep)
    if cols.size == 0:
        return cols.astype(np.int64, copy=False)

    # Deterministic ordering: score descending, then library row ascending.
    order = np.lexsort((cols, -scores[cols]))
    cols = cols[order]

    top_k = cfg.get("search_top_k")
    if top_k is not None and cols.size > int(top_k):
        cols = cols[: int(top_k)]

    return cols.astype(np.int64, copy=False)


def _empty_search_arrays(with_matches: bool):
    rows = np.empty(0, dtype=np.int64)
    cols = np.empty(0, dtype=np.int64)
    scores = np.empty(0, dtype=np.float64)
    if with_matches:
        return rows, cols, scores, np.empty(0, dtype=np.int32)
    return rows, cols, scores


def _search_batch_task_entropy(row_indices):
    """Score and sparsify one batch of entropy query rows inside a worker."""
    refs = _G_REFS
    cfg = _G_CFG
    rows_out = []
    cols_out = []
    scores_out = []

    for row_idx in row_indices:
        row_idx = int(row_idx)
        if (
            cfg.get("search_require_precursor", False)
            and not np.isfinite(float(refs.precursor_mz[row_idx]))
        ):
            continue

        _, row_scores = _row_task_entropy(row_idx)
        dummy_matches = np.empty(0, dtype=np.int32)
        cols = _select_search_hit_columns(row_scores, dummy_matches, cfg)
        if cols.size == 0:
            continue
        rows_out.append(np.full(cols.size, row_idx, dtype=np.int64))
        cols_out.append(cols)
        scores_out.append(row_scores[cols].astype(np.float64, copy=False))

    if not rows_out:
        rows, cols, scores = _empty_search_arrays(with_matches=False)
        return len(row_indices), rows, cols, scores

    return (
        len(row_indices),
        np.concatenate(rows_out),
        np.concatenate(cols_out),
        np.concatenate(scores_out),
    )


def _search_batch_task_cosine(row_indices):
    """Score and sparsify one batch of cosine query rows inside a worker."""
    refs = _G_REFS
    cfg = _G_CFG
    rows_out = []
    cols_out = []
    scores_out = []
    matches_out = []

    for row_idx in row_indices:
        row_idx = int(row_idx)
        if (
            cfg.get("search_require_precursor", False)
            and not np.isfinite(float(refs.precursor_mz[row_idx]))
        ):
            continue

        _, row_scores, row_matches = _row_task_cosine(row_idx)
        cols = _select_search_hit_columns(row_scores, row_matches, cfg)
        if cols.size == 0:
            continue
        rows_out.append(np.full(cols.size, row_idx, dtype=np.int64))
        cols_out.append(cols)
        scores_out.append(row_scores[cols].astype(np.float64, copy=False))
        matches_out.append(row_matches[cols].astype(np.int32, copy=False))

    if not rows_out:
        rows, cols, scores, matches = _empty_search_arrays(with_matches=True)
        return len(row_indices), rows, cols, scores, matches

    return (
        len(row_indices),
        np.concatenate(rows_out),
        np.concatenate(cols_out),
        np.concatenate(scores_out),
        np.concatenate(matches_out),
    )


def _combine_entropy_search_batches(batch_results):
    nonempty = [result for result in batch_results if result[1].size > 0]
    if not nonempty:
        return _empty_search_arrays(with_matches=False)
    return (
        np.concatenate([result[1] for result in nonempty]),
        np.concatenate([result[2] for result in nonempty]),
        np.concatenate([result[3] for result in nonempty]),
    )


def _combine_cosine_search_batches(batch_results):
    nonempty = [result for result in batch_results if result[1].size > 0]
    if not nonempty:
        return _empty_search_arrays(with_matches=True)
    return (
        np.concatenate([result[1] for result in nonempty]),
        np.concatenate([result[2] for result in nonempty]),
        np.concatenate([result[3] for result in nonempty]),
        np.concatenate([result[4] for result in nonempty]),
    )
