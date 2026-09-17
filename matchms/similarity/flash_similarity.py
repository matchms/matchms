"""Indexed spectral similarities with collection-native preprocessing.

A library index stores globally sorted peaks for tolerance-window lookups and
spectrum-major arrays for normalization and peak identity. Searches return dense
``Scores`` with query rows and library columns. Numerical kernels release the
GIL and can process disjoint query blocks using shared, read-only index arrays.
"""
from __future__ import annotations
import operator
import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from matchms.scores import Scores
from matchms.typing import SpectrumType
from ._flash_cosine import cosine_rows
from ._flash_entropy import entropy_rows
from ._flash_prepared import PreparedSpectra, empty_prepared, pack_native
from .base_similarity import BaseSimilarity
from .default_parameters import (
    DEFAULT_DTYPE,
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_index import FlashIndex, config_from_settings
from .flash_index import build_index as _build_index


_MATCHING_MODES = {"fragment": 0, "neutral_loss": 1, "hybrid": 2}
_MAX_SYMMETRIC_PPM = 2_000_000


def _nonnegative(value: float, name: str) -> float:
    """Validate a finite, nonnegative numerical parameter."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _worker_count(n_jobs: int | None, n_queries: int) -> int:
    """Resolve job counts, bounded by available CPUs and query rows.

    Zero and None select one worker; -1 selects all available CPUs; -2 leaves
    one CPU unused. Negative counts are resolved before limiting by query count.
    """
    if n_jobs is None:
        n_jobs = 1
    if isinstance(n_jobs, (bool, np.bool_)):
        raise TypeError("n_jobs must be an integer, not bool.")
    n_jobs = operator.index(n_jobs)
    cpus = os.cpu_count() or 1
    workers = max(1, cpus + 1 + n_jobs) if n_jobs < 0 else max(1, n_jobs)
    return min(workers, cpus, max(1, n_queries))


def _run_rows(
    run: Callable[[int, int], None],
    n_queries: int,
    n_jobs: int | None,
    progress_bar: bool,
    descriptor: str,
) -> None:
    """Run a row kernel serially or in nonoverlapping threaded blocks.

    Each invocation owns its scratch arrays. The first row is evaluated in the
    calling thread before launching workers so that initial JIT compilation is
    not concurrent. Exceptions in workers are propagated to the caller.
    """
    workers = _worker_count(n_jobs, n_queries)
    if n_queries == 0:
        return
    with tqdm(total=n_queries, desc=descriptor, disable=not progress_bar) as progress:
        if workers == 1 and not progress_bar:
            run(0, n_queries)
            return
        run(0, 1)
        progress.update(1)
        if n_queries == 1:
            return
        block_size = max(1, (n_queries - 1 + 4 * workers - 1) // (4 * workers))
        blocks = [
            (start, min(start + block_size, n_queries))
            for start in range(1, n_queries, block_size)
        ]
        if workers == 1:
            for start, stop in blocks:
                run(start, stop)
                progress.update(stop - start)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run, start, stop): stop - start
                    for start, stop in blocks
                }
                for future in as_completed(futures):
                    future.result()
                    progress.update(futures[future])


class _BaseFlashSimilarity(BaseSimilarity):
    """Shared preparation, persistence, and dense-search interface.

    Subclasses provide ``_weighing_type`` and ``_score_prepared``. Preprocessing
    settings describe an index's contents; matching tolerance, identity gates,
    worker count, and selected output fields are search-time choices.

    Parameters
    ----------
    matching_mode
        ``"fragment"``, ``"neutral_loss"``, or ``"hybrid"``. Neutral losses are
        ``precursor_mz - fragment_mz``. The hybrid assignment rule depends on the
        similarity class.
    tolerance
        Inclusive tolerance for matching coordinates, in Da unless ``use_ppm``
        is True. In loss mode, the tolerance applies to loss coordinates.
    use_ppm
        Use ``abs(a - b) <= tolerance * 1e-6 * (a + b) / 2`` instead of Da.
    intensity_power
        Exponent applied to intensities for cosine. Entropy uses its own
        entropy-dependent weighting rather than this power.
    remove_precursor
        Exclude peaks with ``m/z > precursor_mz + offset_to_precursor`` during
        preparation. Missing precursor handling follows the collection cleaner.
    offset_to_precursor
        Signed Da offset for the upper peak cutoff. The boundary is retained.
    noise_cutoff
        Minimum intensity relative to the maximum remaining intensity after
        precursor filtering. Set to zero or None to disable noise filtering.
    normalize_to_half
        Normalize prepared intensities to sum to 0.5. Enabled by default in
        ``FlashEntropy``; required for the usual entropy-similarity scale.
    merge_within
        Optional within-spectrum merge distance, in Da. Zero disables merging.
        Neither scorer requires merging to resolve overlapping matching windows.
    identity_precursor_tolerance
        Optional precursor filter on scored pairs. When set, both precursors
        must be finite and match within this tolerance; otherwise the pair is zero.
    identity_use_ppm
        Interpret the precursor filter in symmetric ppm rather than Da.
    dtype
        Float32 or float64 for prepared peaks and score output. Lower precision
        can change matching decisions at tolerance boundaries.

    Notes
    -----
    Complete dense output requires storage proportional to the number of query
    spectra times the number of library spectra. Each worker also needs its own
    scratch arrays. Hybrid matching uses additional state proportional to the
    library peak count and query size. Use ``n_jobs=1`` to restrict concurrency.
    """

    is_commutative = True
    _weighing_type = "cosine"

    def __init__(
        self,
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        intensity_power: float = DEFAULT_INTENSITY_POWER,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        normalize_to_half: bool = False,
        merge_within: float = 0.0,
        identity_precursor_tolerance: float | None = None,
        identity_use_ppm: bool = False,
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        if matching_mode not in _MATCHING_MODES:
            raise ValueError("matching_mode must be 'fragment', 'neutral_loss', or 'hybrid'.")
        self.matching_mode = matching_mode
        self.tolerance = _nonnegative(tolerance, "tolerance")
        self.use_ppm = bool(use_ppm)
        if self.use_ppm and self.tolerance >= _MAX_SYMMETRIC_PPM:
            raise ValueError("Symmetric ppm tolerance must be < 2,000,000.")
        self.intensity_power = _nonnegative(intensity_power, "intensity_power")
        self.remove_precursor = bool(remove_precursor)
        self.offset_to_precursor = float(offset_to_precursor)
        if not np.isfinite(self.offset_to_precursor):
            raise ValueError("offset_to_precursor must be finite.")
        self.noise_cutoff = (
            None if noise_cutoff is None else _nonnegative(noise_cutoff, "noise_cutoff")
        )
        if self.noise_cutoff is not None and self.noise_cutoff > 1:
            raise ValueError("noise_cutoff must be <= 1.")
        self.normalize_to_half = bool(normalize_to_half)
        self.merge_within = _nonnegative(merge_within, "merge_within")
        self.identity_precursor_tolerance = (
            None if identity_precursor_tolerance is None
            else _nonnegative(identity_precursor_tolerance, "identity_precursor_tolerance")
        )
        self.identity_use_ppm = bool(identity_use_ppm)
        if (
            self.identity_use_ppm
            and self.identity_precursor_tolerance is not None
            and self.identity_precursor_tolerance >= _MAX_SYMMETRIC_PPM
        ):
            raise ValueError("Symmetric identity ppm tolerance must be < 2,000,000.")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64.")
        self.score_datatype = (
            np.dtype([("score", self.dtype), ("matches", np.int32)])
            if self.score_fields == ("score", "matches") else self.dtype
        )

    def to_dict(self) -> dict:
        """Return JSON-compatible constructor parameters, excluding runtime state."""
        names = (
            "matching_mode", "tolerance", "use_ppm", "intensity_power",
            "remove_precursor", "offset_to_precursor", "noise_cutoff",
            "normalize_to_half", "merge_within", "identity_precursor_tolerance",
            "identity_use_ppm",
        )
        return {
            "__Similarity__": type(self).__name__,
            **{name: getattr(self, name) for name in names},
            "dtype": self.dtype.name,
        }

    def _settings(self) -> tuple:
        """Return the settings that determine prepared peak values."""
        return (
            self._weighing_type, self.intensity_power, self.remove_precursor,
            self.offset_to_precursor, self.noise_cutoff or 0.0,
            self.normalize_to_half, self.merge_within, self.dtype.str,
        )

    def prepare_queries(self, spectra: Sequence[SpectrumType]) -> PreparedSpectra:
        """Return validated, packed peaks without modifying the input spectra.

        SpectraCollection inputs are processed through the native collection
        cleaner. Other iterables are converted to a SpectraCollection first,
        using its default m/z precision. An empty input is supported.

        The result can be reused with ``search_prepared`` as long as the scorer's
        preprocessing parameters remain unchanged.
        """
        from matchms.spectra_collection import SpectraCollection
        from .flash_utils import _prepare_collection

        if not isinstance(spectra, SpectraCollection):
            spectra = list(spectra)
            if not spectra:
                return empty_prepared(self.dtype, self._settings())
            spectra = SpectraCollection(spectra)
        if len(spectra) == 0:
            return empty_prepared(self.dtype, self._settings())
        prepared = _prepare_collection(
            spectra,
            intensity_power=self.intensity_power,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
            normalize_to_half=self.normalize_to_half,
            merge_within_da=self.merge_within,
            weighing_type=self._weighing_type,
            compute_l2_norm=self._weighing_type == "cosine",
            dtype=self.dtype,
        )
        return pack_native(prepared, self.dtype, self._settings())

    def build_index(self, spectra: Sequence[SpectrumType]) -> FlashIndex:
        """Prepare reference spectra and build a reusable library index.

        The index records preprocessing settings and collection precision.
        Loss modes additionally store a neutral-loss index. An existing
        loss-capable index can also be used for fragment-only searches.
        """
        from matchms.spectra_collection import SpectraCollection

        if not isinstance(spectra, SpectraCollection):
            spectra = list(spectra)
            if spectra:
                spectra = SpectraCollection(spectra)
        metadata = {}
        if hasattr(spectra, "mz_precision"):
            metadata["mz_precision"] = float(spectra.mz_precision)
        if hasattr(spectra, "fragments"):
            metadata["fragment_backend"] = type(spectra.fragments).__name__
        return self.build_index_prepared(self.prepare_queries(spectra), metadata=metadata)

    def build_index_prepared(
        self, prepared: PreparedSpectra, *, metadata: dict | None = None,
    ) -> FlashIndex:
        """Build an index from compatible packed peaks without reprocessing them."""
        self._check_prepared(prepared)
        return _build_index(
            prepared, self.matching_mode, self._weighing_type, metadata=metadata,
        )

    def _check_prepared(self, prepared: PreparedSpectra) -> None:
        """Check preparation compatibility without rescanning the peak arrays."""
        if not isinstance(prepared, PreparedSpectra) or prepared.settings != self._settings():
            raise ValueError("Prepared spectra do not match this scorer's preprocessing settings.")

    def _check_index(self, library_index: FlashIndex) -> None:
        """Validate preprocessing settings and the required index capabilities."""
        if not isinstance(library_index, FlashIndex):
            raise TypeError(
                "library_index must be a FlashIndex; use search(query_spectra, library_index)."
            )
        expected = config_from_settings(self._settings())
        actual = dict(library_index.config)
        if "merge_within" not in actual and "merge_within_da" in actual:
            actual["merge_within"] = actual["merge_within_da"]
        actual["dtype"] = library_index.dtype.str
        if "noise_cutoff" in actual:
            actual["noise_cutoff"] = actual["noise_cutoff"] or 0.0
        different = [
            name for name, value in expected.items()
            if name not in actual or actual[name] != value
        ]
        if different:
            raise ValueError(
                "FlashIndex preprocessing configuration differs for: "
                + ", ".join(different) + ". Rebuild with matching preprocessing."
            )
        if self.matching_mode != "fragment" and not library_index.has_neutral_loss_index:
            raise ValueError(
                "This matching mode requires a neutral-loss index. Build a "
                "hybrid or neutral_loss index; search does not rebuild it."
            )
        if self._weighing_type == "cosine" and not library_index.has_l2_norms:
            raise ValueError("Cosine search requires an index containing L2 norms.")

    def search(
        self,
        query_spectra: Sequence[SpectrumType],
        library_index: FlashIndex,
        *,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ) -> Scores:
        """Compare queries with an already indexed spectral library.

        Parameters
        ----------
        query_spectra
            Query spectra as a SpectraCollection or an iterable of spectra.
        library_index
            Compatible index returned by ``build_index`` or ``load_index``.
        score_fields
            Fields to return. None selects every available field.
        progress_bar
            Display query-processing progress.
        n_jobs
            Number of worker threads. One selects serial execution; -1 selects
            available CPUs. Index arrays are shared between threads.

        Returns
        -------
        Scores
            Dense array fields with shape ``(n_queries, library_index.n_specs)``.
            Scoring is the same as ``matrix(query_spectra, library_spectra)``;
            reference preprocessing and index construction are not repeated.
        """
        self._check_index(library_index)
        self._resolve_score_fields(score_fields)
        return self.search_prepared(
            self.prepare_queries(query_spectra), library_index,
            score_fields=score_fields, progress_bar=progress_bar, n_jobs=n_jobs,
        )

    def search_prepared(
        self,
        query_spectra: PreparedSpectra,
        library_index: FlashIndex,
        *,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = False,
        n_jobs: int = 1,
    ) -> Scores:
        """Search previously prepared queries, returning query-by-library scores.

        Use ``prepare_queries`` to construct ``query_spectra``. Configuration
        checks apply to both the prepared queries and the index. Index array
        views and dtype conversions are cached and reused across calls.
        """
        self._check_index(library_index)
        self._check_prepared(query_spectra)
        fields = self._resolve_score_fields(score_fields)
        return self._score_prepared(query_spectra, library_index, fields, progress_bar, n_jobs)

    @abstractmethod
    def _score_prepared(
        self,
        queries: PreparedSpectra,
        library_index: FlashIndex,
        fields: tuple[str, ...],
        progress_bar: bool,
        n_jobs: int,
    ) -> Scores:
        """Allocate output and dispatch the subclass's numerical row kernel."""
        raise NotImplementedError

    def _prepare_matrix_inputs(self, spectra_1, spectra_2) -> tuple:
        """Prepare each distinct input once; None requests self-comparison."""
        first = self.prepare_queries(spectra_1)
        if spectra_2 is None:
            return first, first, True
        second = first if spectra_2 is spectra_1 else self.prepare_queries(spectra_2)
        return first, second, False

    def _optimize_matrix_orientation(self, refs, queries, is_symmetric) -> tuple:
        """Return a smaller query side and a flag requesting output transposition.

        Entropy uses this orientation. Cosine retains the requested direction
        because equal-weight greedy choices can depend on candidate ordering.
        """
        if is_symmetric or not self.is_commutative or refs.n_specs <= queries.n_specs:
            return refs, queries, False
        return queries, refs, True

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ) -> Scores:
        """Calculate dense all-pairs scores in the requested input orientation.

        Rows correspond to ``spectra_1`` and columns to ``spectra_2``. If the
        second input is None, compare the first input against itself. Both
        preparation and index construction are included in this call.

        ``score_fields`` selects output fields; ``n_jobs`` controls worker
        threads as in ``search``. Cosine always retains the requested scoring
        direction. Entropy can index the larger side and transpose output views
        without copying a complete matrix. Self-comparison reuses preparation.
        """
        self._resolve_score_fields(score_fields)
        first, second, is_symmetric = self._prepare_matrix_inputs(spectra_1, spectra_2)
        transpose = False
        if self._weighing_type == "entropy":
            first, second, transpose = self._optimize_matrix_orientation(first, second, is_symmetric)
        index = self.build_index_prepared(second)
        scores = self.search_prepared(
            first, index, score_fields=score_fields,
            progress_bar=progress_bar, n_jobs=n_jobs,
        )
        if transpose:
            return Scores({field: array.T for field, array in scores._data.items()})
        return scores

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """Return one scalar or structured score using the indexed scoring kernel.

        The same preparation and matching rules apply as in ``matrix``. Cosine
        returns fields ``score`` and ``matches``; entropy returns a scalar score.
        """
        result = self.matrix([spectrum_1], [spectrum_2], progress_bar=False, n_jobs=1)
        if self.score_fields == ("score",):
            return np.asarray(result._data["score"][0, 0], dtype=self.score_datatype)
        return np.asarray(
            (result._data["score"][0, 0], result._data["matches"][0, 0]),
            dtype=self.score_datatype,
        )

    def save_index(
        self, index: FlashIndex, filename: str | Path, *, overwrite: bool = True,
    ) -> None:
        """Save an index after checking preparation settings and capabilities."""
        self._check_index(index)
        index.save(filename, overwrite=overwrite)

    def load_index(self, filename: str | Path) -> FlashIndex:
        """Load a compatible index and initialize its read-only kernel views."""
        index = FlashIndex.load(filename)
        self._check_index(index)
        if self._weighing_type == "entropy":
            index.entropy_data()
        else:
            index.cosine_data()
        return index


class CosineFlash(_BaseFlashSimilarity):
    """Cosine similarity from indexed peaks with greedy conflict resolution.

    Independent peak matches are accumulated directly. If candidates compete
    for a peak, candidates for that spectrum pair are sorted by intensity
    product and assigned one-to-one. A small additive preference for fragment
    candidates preserves the Flash cosine assignment convention.

    ``matching_mode="fragment"`` matches fragment coordinates. ``"neutral_loss"``
    matches loss coordinates only. ``"hybrid"`` combines both candidate sets
    before assignment, providing modified cosine scoring. When the precursor
    difference is within tolerance, hybrid scoring uses direct candidates only.
    Missing precursors prevent loss matches but not direct fragment matches.

    Results contain ``score`` and ``matches``. The latter counts accepted
    one-to-one assignments, not candidate pairs. Request ``score_fields=("score",)``
    to avoid allocating a dense match-count output. Numerical kernels are compiled
    with Numba; no separate score-only implementation is required.

    See :class:`_BaseFlashSimilarity` for preparation and search parameters.
    """

    score_fields = ("score", "matches")
    score_datatype = np.dtype([("score", np.float64), ("matches", np.int32)])

    def _score_prepared(self, queries, library_index, fields, progress_bar, n_jobs) -> Scores:
        """Calculate cosine fields, omitting the count array when not requested."""
        library = library_index.cosine_data()
        scores = np.empty((queries.n_specs, library.n_specs), dtype=self.dtype)
        counts = np.empty(scores.shape if "matches" in fields else (0, 0), dtype=np.int32)
        mode = _MATCHING_MODES[self.matching_mode]
        gate = -1.0 if self.identity_precursor_tolerance is None else self.identity_precursor_tolerance

        def run(start, stop):
            """Fill an independent query-row block."""
            cosine_rows(
                scores, counts, queries.spec_offsets, queries.spec_mz, queries.spec_int,
                queries.precursor_mz, queries.spec_l2,
                library.peaks_mz, library.peaks_int, library.peaks_spec_idx,
                library.nl_mz, library.nl_spec_idx, library.nl_product_idx,
                library.precursor_mz, library.spec_l2,
                self.tolerance, self.use_ppm, mode, gate, self.identity_use_ppm, start, stop,
            )

        _run_rows(run, queries.n_specs, n_jobs, progress_bar, f"{type(self).__name__} ({self.matching_mode})")
        arrays = {"score": scores, "matches": counts}
        return Scores({field: arrays[field] for field in fields})


class FlashEntropy(_BaseFlashSimilarity):
    """Entropy similarity accumulated directly from matching indexed peaks.

    Fragment matches are consumed one-to-one in ascending m/z order. Neutral-loss
    matches use ascending ``precursor_mz - fragment_mz`` coordinates. Hybrid
    scoring first accepts fragment matches, then loss matches involving only
    unused peaks. This differs from cosine's joint intensity-product assignment.

    Every physical peak contributes at most once for each spectrum pair, including
    spectra with overlapping tolerance windows. Missing precursors give zero in
    neutral-loss mode and fragment-only hybrid scores, provided preprocessing has
    retained the spectra. Library entropy terms are cached during index building.

    The default entropy weighting and half normalization produce scores in
    [0, 1], up to roundoff. Disabling half normalization changes that scale.
    The returned field is ``score``.

    This implementation was insipred by Flash entropy similarity (Li & Fiehn, 2023).

    See :class:`_BaseFlashSimilarity` for preparation and search parameters.
    """

    _weighing_type = "entropy"
    score_fields = ("score",)
    score_datatype = np.float64

    def __init__(
        self,
        matching_mode: str = "fragment",
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        use_ppm: bool = False,
        intensity_power: float = DEFAULT_INTENSITY_POWER,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        normalize_to_half: bool = True,
        merge_within: float = 0.0,
        identity_precursor_tolerance: float | None = None,
        identity_use_ppm: bool = False,
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        super().__init__(
            matching_mode=matching_mode, tolerance=tolerance, use_ppm=use_ppm,
            intensity_power=intensity_power, remove_precursor=remove_precursor,
            offset_to_precursor=offset_to_precursor, noise_cutoff=noise_cutoff,
            normalize_to_half=normalize_to_half, merge_within=merge_within,
            identity_precursor_tolerance=identity_precursor_tolerance,
            identity_use_ppm=identity_use_ppm, dtype=dtype,
        )

    def _score_prepared(self, queries, library_index, fields, progress_bar, n_jobs) -> Scores:
        """Dispatch entropy rows using the index's cached entropy contributions."""
        library = library_index.entropy_data()
        scores = np.empty((queries.n_specs, library.n_specs), dtype=self.dtype)
        mode = _MATCHING_MODES[self.matching_mode]
        gate = -1.0 if self.identity_precursor_tolerance is None else self.identity_precursor_tolerance

        def run(start, stop):
            """Fill an independent query-row block."""
            entropy_rows(
                scores, queries.spec_offsets, queries.spec_mz, queries.spec_int,
                queries.precursor_mz, library.fragment, library.neutral_loss,
                library.precursor_mz, library.n_peaks,
                self.tolerance, self.use_ppm, mode, gate, self.identity_use_ppm, start, stop,
            )

        _run_rows(run, queries.n_specs, n_jobs, progress_bar, f"{type(self).__name__} ({self.matching_mode})")
        return Scores({"score": scores})
