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


class Cosine(CosineFlash):
    """Compare mass spectra using cosine similarity and matched-peak counts.

    By default, indexed peaks provide candidate matches within ``tolerance``.
    Independent matches are summed directly; competing matches are resolved by
    descending intensity-product greedy assignment. ``pair``, ``matrix``, and
    ``search`` share this preparation and assignment rule.

    Set ``use_hungarian=True`` for optimal peak assignment in ``pair`` and
    ``matrix``. This alternative does not provide indexed searches. For a simple
    spectrum-pair implementation, see :class:`~matchms.similarity.CosineGreedy`.

    Parameters
    ----------
    tolerance
        Maximum m/z difference for a peak match. The boundary is inclusive.
    intensity_power
        Exponent applied to peak intensities before cosine scoring.
    use_hungarian
        Use Hungarian rather than greedy peak assignment. This selects a
        different algorithm, not a different preprocessing pipeline.
    noise_cutoff
        Relative intensity cutoff after precursor-region removal. Set to zero
        or None to disable noise filtering.
    remove_precursor
        Remove peaks above ``precursor_mz + offset_to_precursor``.
    offset_to_precursor
        Signed Da offset defining the precursor-region cutoff.
    use_ppm
        Interpret the matching tolerance as symmetric ppm rather than Da.
        Supported by indexed greedy scoring only.
    merge_within
        Optional within-spectrum merge distance in Da. Zero disables merging.
        Supported by indexed greedy scoring only.
    dtype
        Float32 or float64 for prepared peaks and returned score values.

    Notes
    -----
    Results contain ``score`` and ``matches``. To omit the dense count output,
    pass ``score_fields=("score",)`` to ``matrix`` or ``search``. ``search`` uses
    a previously built :class:`~matchms.similarity.flash_index.FlashIndex` and
    returns query rows and library columns. For reproducible indexed workflows,
    store the preprocessing settings together with the index.
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
