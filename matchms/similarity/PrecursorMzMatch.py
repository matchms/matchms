from typing import Optional, Sequence
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


class PrecursorMzMatch(BaseSimilarityWithSparse):
    """Return True if spectra match in precursor m/z, and False otherwise.

    The match within tolerance can be calculated based on an absolute m/z
    difference (``tolerance_type="Dalton"``) or based on a relative
    difference in ppm (``tolerance_type="ppm"``).

    Example to calculate scores between 2 pairs of spectra and inspect the
    resulting score matrix:

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import PrecursorMzMatch

        spectrum_1 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"id": "1", "precursor_mz": 100},
        )
        spectrum_2 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"id": "2", "precursor_mz": 110},
        )
        spectrum_3 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"id": "3", "precursor_mz": 103},
        )
        spectrum_4 = Spectrum(
            mz=np.array([]),
            intensities=np.array([]),
            metadata={"id": "4", "precursor_mz": 111},
        )

        spectra_1 = [spectrum_1, spectrum_2]
        spectra_2 = [spectrum_3, spectrum_4]

        similarity = PrecursorMzMatch(tolerance=5.0, tolerance_type="Dalton")
        scores = similarity.matrix(spectra_1, spectra_2)

        score_array = scores.to_array()

        for i, spectrum_1 in enumerate(spectra_1):
            for j, spectrum_2 in enumerate(spectra_2):
                print(
                    f"Precursor m/z match between {spectrum_1.get('id')} and "
                    f"{spectrum_2.get('id')} is {bool(score_array[i, j])}"
                )

    Should output

    .. testoutput::

        Precursor m/z match between 1 and 3 is True
        Precursor m/z match between 1 and 4 is False
        Precursor m/z match between 2 and 3 is False
        Precursor m/z match between 2 and 4 is True
    """

    is_commutative = True
    score_datatype = bool
    score_fields = ("score",)

    def __init__(self, tolerance: float = 0.1, tolerance_type: str = "Dalton"):
        """
        Parameters
        ----------
        tolerance
            Specify tolerance below which two precursor m/z values are counted as match.
        tolerance_type
            Choose between fixed tolerance in Dalton (``"Dalton"``) or a relative
            difference in ppm (``"ppm"``).
        """
        self.tolerance = tolerance
        assert tolerance_type in ["Dalton", "ppm"], "Expected type from ['Dalton', 'ppm']"
        self.type = tolerance_type

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType):
        """Compare precursor m/z between two spectra.

        Parameters
        ----------
        spectrum_1
            First spectrum.
        spectrum_2
            Second spectrum.
        """
        precursor_mz_1 = spectrum_1.get("precursor_mz")
        precursor_mz_2 = spectrum_2.get("precursor_mz")
        assert precursor_mz_1 is not None and precursor_mz_2 is not None, "Missing precursor m/z."

        if self.type == "Dalton":
            score = abs(precursor_mz_1 - precursor_mz_2) <= self.tolerance
            return np.asarray(score, dtype=self.score_datatype)

        mean_mz = (precursor_mz_1 + precursor_mz_2) / 2
        ppm_difference = abs(precursor_mz_1 - precursor_mz_2) / mean_mz * 1e6
        score = ppm_difference <= self.tolerance
        return np.asarray(score, dtype=self.score_datatype)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        score_fields: Optional[Sequence[str]] = None,
        progress_bar: bool = True,
    ) -> Scores:
        """Compare precursor m/z between all spectra in `spectra_1` and `spectra_2`.

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

        Returns
        -------
        Scores
            Dense score matrix as a `Scores` object.
        """
        del progress_bar  # not used in optimized implementation

        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError(
                "PrecursorMzMatch.matrix() supports only score_fields=('score',)."
            )

        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)

        precursor_mz_1 = self._collect_precursor_mz(spectra_1)
        precursor_mz_2 = self._collect_precursor_mz(spectra_2)

        if is_symmetric and self.type == "Dalton":
            rows, cols, scores = number_matching_symmetric(precursor_mz_1, self.tolerance)
        elif is_symmetric and self.type == "ppm":
            rows, cols, scores = number_matching_symmetric_ppm(precursor_mz_1, self.tolerance)
        elif self.type == "Dalton":
            rows, cols, scores = number_matching(precursor_mz_1, precursor_mz_2, self.tolerance)
        else:
            rows, cols, scores = number_matching_ppm(precursor_mz_1, precursor_mz_2, self.tolerance)

        score_array = np.zeros((len(precursor_mz_1), len(precursor_mz_2)), dtype=self.score_datatype)
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
        """Compare precursor m/z and return sparse scores.

        This method uses the optimized precursor-m/z matching functions when no
        explicit indices are provided. If explicit `idx_row` and `idx_col` are
        given, it falls back to the generic sparse implementation from
        `BaseSimilarityWithSparse`.
        """
        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError(
                "PrecursorMzMatch.sparse_matrix() supports only score_fields=('score',)."
            )

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

        precursor_mz_1 = self._collect_precursor_mz(spectra_1)
        precursor_mz_2 = self._collect_precursor_mz(spectra_2)

        if is_symmetric and self.type == "Dalton":
            rows, cols, scores = number_matching_symmetric(precursor_mz_1, self.tolerance)
        elif is_symmetric and self.type == "ppm":
            rows, cols, scores = number_matching_symmetric_ppm(precursor_mz_1, self.tolerance)
        elif self.type == "Dalton":
            rows, cols, scores = number_matching(precursor_mz_1, precursor_mz_2, self.tolerance)
        else:
            rows, cols, scores = number_matching_ppm(precursor_mz_1, precursor_mz_2, self.tolerance)

        scores = scores.astype(self.score_datatype, copy=False)

        if score_filter is not None:
            keep = np.array([bool(score_filter(np.asarray(v, dtype=self.score_datatype))) for v in scores], dtype=bool)
            rows = rows[keep]
            cols = cols[keep]
            scores = scores[keep]

        sparse = coo_array(
            (scores, (rows, cols)),
            shape=(len(precursor_mz_1), len(precursor_mz_2)),
            dtype=self.score_datatype,
        )
        sparse.eliminate_zeros()
        return Scores({"score": sparse})

    @staticmethod
    def _collect_precursor_mz(spectra: Sequence[SpectrumType]) -> np.ndarray:
        """Collect precursor m/z values from spectra."""
        precursor_mz = []
        for spectrum in spectra:
            value = spectrum.get("precursor_mz")
            assert value is not None, "Missing precursor m/z."
            precursor_mz.append(value)
        return np.asarray(precursor_mz)
