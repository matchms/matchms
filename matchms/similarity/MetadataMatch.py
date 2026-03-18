import logging
from collections import defaultdict
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.sparse import coo_array
from matchms.Scores import Scores
from matchms.similarity.spectrum_similarity_functions import (
    number_matching,
    number_matching_ppm,
    number_matching_symmetric,
    number_matching_symmetric_ppm,
)
from matchms.typing import ScoreFilter, SpectrumType
from .BaseSimilarity import BaseSimilarityWithSparse


logger = logging.getLogger("matchms")

_MISSING = object()


class MetadataMatch(BaseSimilarityWithSparse):
    """Return True if metadata entries of a specified field match between two spectra.

    This is supposed to be used to compare a wide range of possible metadata entries and
    use this to later select related or similar spectra.

    Matching can be done by:

    - exact equality (``matching_type="equal_match"``)
    - numerical difference within a tolerance (``matching_type="difference"``)

    For numerical differences, the tolerance can be interpreted as:

    - absolute difference in Dalton / raw units (``tolerance_type="Dalton"``)
    - relative difference in ppm (``tolerance_type="ppm"``)

    Example to calculate scores between 2 pairs of spectra and inspect the score matrix

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import MetadataMatch

        spectrum_1 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"instrument_type": "orbitrap", "id": 1},
        )
        spectrum_2 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"instrument_type": "qtof", "id": 2},
        )
        spectrum_3 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"instrument_type": "qtof", "id": 3},
        )
        spectrum_4 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"instrument_type": "orbitrap", "id": 4},
        )

        spectra_1 = [spectrum_1, spectrum_2]
        spectra_2 = [spectrum_3, spectrum_4]

        similarity = MetadataMatch(field="instrument_type")
        scores = similarity.matrix(spectra_1, spectra_2)

        score_array = scores.to_array()

        for i, spectrum_1 in enumerate(spectra_1):
            for j, spectrum_2 in enumerate(spectra_2):
                print(
                    f"Metadata match between {spectrum_1.get('id')} and "
                    f"{spectrum_2.get('id')} is {bool(score_array[i, j])}"
                )

    Should output

    .. testoutput::

        Metadata match between 1 and 3 is False
        Metadata match between 1 and 4 is True
        Metadata match between 2 and 3 is True
        Metadata match between 2 and 4 is False
    """

    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = bool
    score_fields = ("score",)

    def __init__(
        self,
        field: str,
        matching_type: str = "equal_match",
        tolerance: float = 0.1,
        tolerance_type: str = "Dalton",
    ):
        """
        Parameters
        ----------
        field
            Specify field name for metadata that should be compared.
        matching_type
            Specify how field entries should be matched. Can be one of
            ``["equal_match", "difference"]``.
            ``"equal_match"``: entries must be exactly equal (default).
            ``"difference"``: entries are considered a match if their numerical
            difference is less than or equal to ``tolerance``.
        tolerance
            Specify tolerance below which two values are counted as match.
            This only applies to numerical values.
        tolerance_type
            Choose between fixed tolerance in Dalton / raw units (``"Dalton"``)
            or a relative difference in ppm (``"ppm"``).
            This only applies when ``matching_type="difference"``.
        """
        self.field = field
        self.tolerance = tolerance

        assert matching_type in ["equal_match", "difference"], "Expected type from ['equal_match', 'difference']"
        self.matching_type = matching_type

        assert tolerance_type in ["Dalton", "ppm"], "Expected type from ['Dalton', 'ppm']"
        self.tolerance_type = tolerance_type

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType):
        """Compare metadata entries between two spectra.

        Parameters
        ----------
        spectrum_1
            First spectrum.
        spectrum_2
            Second spectrum.
        """
        entry_1 = spectrum_1.get(self.field)
        entry_2 = spectrum_2.get(self.field)

        if entry_1 is None or entry_2 is None:
            return np.asarray(False, dtype=self.score_datatype)

        if self.matching_type == "equal_match":
            score = entry_1 == entry_2
            return np.asarray(score, dtype=self.score_datatype)

        if isinstance(entry_1, (int, float)) and isinstance(entry_2, (int, float)):
            if self.tolerance_type == "Dalton":
                score = abs(entry_1 - entry_2) <= self.tolerance
            else:
                mean_value = (entry_1 + entry_2) / 2
                if mean_value == 0:
                    score = entry_1 == entry_2
                else:
                    ppm_difference = abs(entry_1 - entry_2) / mean_value * 1e6
                    score = ppm_difference <= self.tolerance
            return np.asarray(score, dtype=self.score_datatype)

        logger.warning("Non-numerical entry not compatible with 'difference' method")
        return np.asarray(False, dtype=self.score_datatype)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        score_fields: Optional[Sequence[str]] = None,
        progress_bar: bool = True,
    ) -> Scores:
        """Compare metadata entries between all spectra in `spectra_1` and `spectra_2`.

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
            Included for API compatibility. Not used here because this optimized
            implementation does not iterate pairwise in Python.
        """
        del progress_bar  # not used in optimized implementation

        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError("MetadataMatch.matrix() supports only score_fields=('score',).")

        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)

        entries_1 = self._collect_entries(spectra_1)
        entries_2 = self._collect_entries(spectra_2)

        rows, cols, scores = self._find_matching_indices(entries_1, entries_2, is_symmetric)

        score_array = np.zeros((len(entries_1), len(entries_2)), dtype=self.score_datatype)
        score_array[rows, cols] = scores.astype(self.score_datatype, copy=False)
        return Scores({"score": score_array})

    def sparse_matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        idx_row=None,
        idx_col=None,
        score_fields: Optional[Sequence[str]] = None,
        score_filter: Optional[ScoreFilter] = None,
        progress_bar: bool = True,
    ) -> Scores:
        """Compare metadata entries and return sparse scores.

        This method uses optimized metadata matching when no explicit indices are
        provided. If explicit `idx_row` and `idx_col` are given, it falls back to
        the generic sparse implementation from `BaseSimilarityWithSparse`.
        """
        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError("MetadataMatch.sparse_matrix() supports only score_fields=('score',).")

        if idx_row is not None or idx_col is not None:
            return super().sparse_matrix(
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                idx_row=idx_row,
                idx_col=idx_col,
                score_fields=score_fields,
                score_filter=score_filter,
                progress_bar=progress_bar,
            )

        del progress_bar  # not used in optimized implementation

        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)

        entries_1 = self._collect_entries(spectra_1)
        entries_2 = self._collect_entries(spectra_2)

        rows, cols, scores = self._find_matching_indices(entries_1, entries_2, is_symmetric)
        scores = scores.astype(self.score_datatype, copy=False)

        if score_filter is not None:
            keep_true = bool(score_filter(np.asarray(True, dtype=self.score_datatype)))
            if not keep_true:
                rows = np.array([], dtype=np.int_)
                cols = np.array([], dtype=np.int_)
                scores = np.array([], dtype=self.score_datatype)

        sparse = coo_array(
            (scores, (rows, cols)),
            shape=(len(entries_1), len(entries_2)),
            dtype=self.score_datatype,
        )
        sparse.eliminate_zeros()
        return Scores({"score": sparse})

    def _collect_entries(self, spectra: Sequence[SpectrumType]) -> np.ndarray:
        """Collect metadata entries for the selected field.

        Missing entries are converted to sentinel values so they can be excluded
        from optimized matrix matching.
        """
        entries = []
        for spectrum in spectra:
            entry = spectrum.get(self.field)

            if entry is None:
                logger.warning("No %s entry found for spectrum.", self.field)
                if self.matching_type == "equal_match":
                    entry = _MISSING
                else:
                    entry = np.nan

            elif self.matching_type == "difference":
                if not isinstance(entry, (int, float)):
                    logger.warning(
                        "Non-numerical entry (%s) not compatible with 'difference' method.",
                        entry,
                    )
                    entry = np.nan

            entries.append(entry)

        if self.matching_type == "equal_match":
            return np.asarray(entries, dtype=object)
        return np.asarray(entries, dtype=float)

    def _find_matching_indices(
        self,
        entries_1: np.ndarray,
        entries_2: np.ndarray,
        is_symmetric: bool,
    ):
        """Find matching indices for optimized matrix / sparse_matrix computation."""
        if self.matching_type == "equal_match":
            if self.tolerance != 0:
                logger.warning("Tolerance is set but will be ignored because 'equal_match' does not use tolerance.")
            if self.tolerance_type != "Dalton":
                logger.warning(
                    "tolerance_type is set but will be ignored because 'equal_match' does not use tolerance."
                )

            rows, cols = self._find_matches_hashmap(entries_1, entries_2)
            scores = np.ones(len(rows), dtype=self.score_datatype)
            return rows, cols, scores

        if self.tolerance_type == "Dalton":
            if is_symmetric:
                return number_matching_symmetric(entries_1, self.tolerance)
            return number_matching(entries_1, entries_2, self.tolerance)

        if is_symmetric:
            return number_matching_symmetric_ppm(entries_1, self.tolerance)
        return number_matching_ppm(entries_1, entries_2, self.tolerance)

    @staticmethod
    def _find_matches_hashmap(entries_1: np.ndarray, entries_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lookup = defaultdict(list)
        for i, entry in enumerate(entries_1):
            if entry is not _MISSING:
                lookup[entry].append(i)

        rows = []
        cols = []

        for j, entry in enumerate(entries_2):
            if entry in lookup:
                match_indices = lookup[entry]
                rows.extend(match_indices)
                cols.extend([j] * len(match_indices))

        rows = np.asarray(rows, dtype=np.int_)
        cols = np.asarray(cols, dtype=np.int_)

        return rows, cols
