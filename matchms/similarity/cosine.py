"""High-level cosine similarity with indexed greedy or Hungarian assignment."""
from collections.abc import Sequence
import numpy as np
from matchms.scores import Scores
from matchms.typing import SpectrumType
from .default_parameters import (
    DEFAULT_DTYPE,
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .flash_index import FlashIndex
from .flash_similarity import CosineFlash


class Cosine(...):
    """Compare mass spectra using cosine similarity.

    Cosine similarity measures the overlap between two spectra after applying
    the configured intensity transformation. Peaks are considered candidate
    matches when their m/z values differ by at most ``tolerance``. Each peak can
    contribute to at most one accepted match.

    By default, matchms uses an indexed greedy implementation. Candidate matches
    that cannot conflict with another assignment are accumulated directly.
    Where multiple candidate matches compete for the same peak, candidates are
    resolved in descending order of their intensity product. The returned
    ``matches`` field is the number of peak pairs accepted by this one-to-one
    assignment.

    The class provides three complementary ways to calculate similarities:

    - :meth:`pair` compares two spectra.
    - :meth:`matrix` calculates a complete similarity matrix between collections
      of spectra.
    - :meth:`build_index` together with :meth:`search` supports repeated queries
      against a fixed reference library without rebuilding the library index for
      every query batch.

    ``matrix`` and the indexed ``search`` path use the same peak preparation and
    matching semantics. ``search(query_spectra, library_index)`` returns query
    spectra as rows and library spectra as columns.

    Set ``use_hungarian=True`` to replace the greedy peak assignment with an
    optimal Hungarian assignment for :meth:`pair` and :meth:`matrix`. Hungarian
    matching is substantially more expensive and does not support persistent
    indexed searches. For a compact pair-oriented greedy implementation, see
    :class:`~matchms.similarity.CosineGreedy`.

    Parameters
    ----------
    tolerance
        Maximum difference between two fragment m/z values for a candidate
        match. The tolerance boundary is inclusive. Interpreted as Da unless
        ``use_ppm=True``.
    intensity_power
        Exponent applied to peak intensities before cosine scoring. The default
        of 1 uses the original intensities; values below 1 reduce the influence
        of very intense peaks.
    use_hungarian
        If False, use the indexed greedy assignment used by the default matchms
        cosine implementation. If True, use Hungarian assignment for ``pair``
        and ``matrix``. Persistent index construction and ``search`` are not
        available with Hungarian matching.
    noise_cutoff
        Remove peaks with intensity below this fraction of the largest remaining
        peak in the spectrum. Set to 0 or None to disable relative-intensity
        filtering.
    remove_precursor
        If True and ``precursor_mz`` is available, remove peaks above
        ``precursor_mz + offset_to_precursor`` before scoring.
    offset_to_precursor
        Signed offset in Da used for precursor-region removal. With the default
        negative value, peaks close to and above the precursor are removed.
    use_ppm
        If True, interpret ``tolerance`` as a symmetric ppm tolerance instead
        of an absolute Da tolerance. Supported by the indexed greedy
        implementation.
    merge_within
        Optional within-spectrum peak-merging distance in Da. Set to 0 to
        disable merging. Supported by the indexed greedy implementation.
    dtype
        Floating-point dtype used for prepared peak intensities and similarity
        scores. Supported values are ``numpy.float32`` and ``numpy.float64``.

    Returns
    -------
    Scores
        ``matrix`` and ``search`` return a
        :class:`~matchms.scores.Scores` object containing the fields ``"score"``
        and ``"matches"`` by default. The ``"score"`` field contains cosine
        similarity values and ``"matches"`` contains the number of accepted
        peak pairs.

    Notes
    -----
    The persistent search workflow separates reference-library preparation from
    query scoring. A library index can therefore be constructed once and reused
    for many independent query batches::

        similarity = Cosine(tolerance=0.01)

        library_index = similarity.build_index(library_spectra)

        scores = similarity.search(
            query_spectra,
            library_index,
            progress_bar=False,
            n_jobs=1,
        )

    The resulting score matrix has shape
    ``(len(query_spectra), len(library_spectra))``.

    Library indices are represented by
    :class:`~matchms.similarity.flash_index.FlashIndex`. The index stores the
    preprocessing configuration required for compatibility checks, so an index
    created with incompatible preparation settings cannot silently be reused by
    another similarity configuration.

    For large dense calculations, storing matched-peak counts can require
    substantial additional memory. If only cosine scores are needed, request
    the score field explicitly::

        scores = similarity.matrix(
            spectra,
            score_fields=("score",),
            progress_bar=False,
            n_jobs=1,
        )

    The same field selection is available for ``search``.

    Examples
    --------
    Compare a single pair of spectra::

        similarity = Cosine(tolerance=0.01)
        result = similarity.pair(spectrum_1, spectrum_2)

        cosine_score = result["score"]
        n_matches = result["matches"]

    Calculate a complete pairwise matrix::

        scores = similarity.matrix(
            spectra,
            progress_bar=False,
            n_jobs=1,
        )

    Build a reusable reference index and search it repeatedly::

        index = similarity.build_index(reference_spectra)

        scores_1 = similarity.search(
            query_batch_1,
            index,
            progress_bar=False,
            n_jobs=1,
        )

        scores_2 = similarity.search(
            query_batch_2,
            index,
            progress_bar=False,
            n_jobs=1,
        )
    """

    _default_mode = "fragment"

    def __init__(
        self,
        tolerance: float = DEFAULT_MZ_TOLERANCE,
        intensity_power: float = DEFAULT_INTENSITY_POWER,
        use_hungarian: bool = False,
        noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
        remove_precursor: bool = True,
        offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
        *,
        use_ppm: bool = False,
        merge_within: float = 0.0,
        dtype: np.dtype = DEFAULT_DTYPE,
    ):
        self.use_hungarian = bool(use_hungarian)
        super().__init__(
            matching_mode=self._default_mode,
            tolerance=tolerance,
            use_ppm=use_ppm,
            intensity_power=intensity_power,
            noise_cutoff=noise_cutoff,
            remove_precursor=remove_precursor,
            offset_to_precursor=offset_to_precursor,
            merge_within=merge_within,
            dtype=dtype,
        )
        if self.use_hungarian and (self.use_ppm or self.merge_within != 0):
            raise ValueError("The Hungarian backend does not expose ppm matching or peak merging.")

    def to_dict(self) -> dict:
        """Return constructor parameters without the underlying index state."""
        names = (
            "tolerance", "intensity_power", "use_hungarian", "noise_cutoff",
            "remove_precursor", "offset_to_precursor", "use_ppm", "merge_within",
        )
        return {
            "__Similarity__": type(self).__name__,
            **{name: getattr(self, name) for name in names},
            "dtype": self.dtype.name,
        }

    def _hungarian(self):
        """Construct the optimal-assignment implementation with shared settings."""
        if self._default_mode == "hybrid":
            from .modified_cosine_hungarian import ModifiedCosineHungarian as Implementation
        else:
            from .cosine_hungarian import CosineHungarian as Implementation
        return Implementation(
            tolerance=self.tolerance,
            intensity_power=self.intensity_power,
            noise_cutoff=self.noise_cutoff,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
        )

    def _require_greedy_index(self) -> None:
        """Reject index operations when optimal assignment has been requested."""
        if self.use_hungarian:
            raise ValueError("Persistent Flash indices/search are unavailable when use_hungarian=True.")

    def _check_index(self, index: FlashIndex) -> None:
        """Check that greedy index use and preprocessing settings are compatible."""
        self._require_greedy_index()
        super()._check_index(index)

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> np.ndarray:
        """Return the cosine score and the number of accepted peak matches."""
        if self.use_hungarian:
            value = self._hungarian().pair(spectrum_1, spectrum_2)
            return np.asarray((value["score"], value["matches"]), dtype=self.score_datatype)
        return super().pair(spectrum_1, spectrum_2)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
        n_jobs: int = -1,
    ) -> Scores:
        """Calculate dense scores with the selected assignment algorithm.

        Rows correspond to ``spectra_1`` and columns to ``spectra_2``. None as
        the second input requests self-comparison. ``n_jobs`` controls query
        workers for the indexed path; Hungarian matrix evaluation is serial.

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
        if self.use_hungarian:
            selected = self._resolve_score_fields(score_fields)
            result = self._hungarian().matrix(
                spectra_1, spectra_2, score_fields=selected, progress_bar=progress_bar,
            )
            return Scores({
                field: result._data[field].astype(self.score_datatype[field], copy=False)
                for field in selected
            })
        return super().matrix(
            spectra_1, spectra_2, score_fields=score_fields,
            progress_bar=progress_bar, n_jobs=n_jobs,
        )

    def build_index(self, spectra) -> FlashIndex:
        """Prepare library spectra and build an index for greedy cosine search."""
        self._require_greedy_index()
        return super().build_index(spectra)

    def build_index_prepared(self, prepared, *, metadata: dict | None = None) -> FlashIndex:
        """Build a greedy-search index from already prepared library spectra."""
        self._require_greedy_index()
        return super().build_index_prepared(prepared, metadata=metadata)
