from collections.abc import Sequence
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]
from matchms.scores import Scores
from matchms.similarity.spectrum_similarity_functions import _process_spectrum_peaks
from matchms.typing import SpectrumType
from .base_similarity import BaseSimilarity
from .cosine_linear_functions import linear_cosine_score, sirius_merge_close_peaks
from .default_parameters import (
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)


class CosineLinear(BaseSimilarity):
    """Calculate 'linear cosine similarity score' between two spectra.

    This implements the CosineLinear similarity from SIRIUS (BOECKER lab), which
    achieves O(n+m) time complexity by requiring spectra to be "well-separated"
    (consecutive peaks more than 2x tolerance apart). A preprocessing step
    (sirius_merge_close_peaks) enforces this invariant by greedily merging close
    peaks in descending intensity order.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import CosineLinear

        reference = Spectrum(mz=np.array([100, 150, 200.]),
                             intensities=np.array([0.7, 0.2, 0.1]))
        query = Spectrum(mz=np.array([100, 140, 190.]),
                         intensities=np.array([0.4, 0.2, 0.1]))

        cosine_linear = CosineLinear(tolerance=0.2)
        score = cosine_linear.pair(reference, query)

        print(f"CosineLinear score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        CosineLinear score is 0.83 with 1 matched peaks

    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]  # type: ignore[assignment]
    score_fields = ("score", "matches")

    def __init__(
            self,
            tolerance: float = DEFAULT_MZ_TOLERANCE,
            mz_power: float = DEFAULT_MZ_POWER,
            intensity_power: float = DEFAULT_INTENSITY_POWER,
            noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
            remove_precursor: bool = False,
            offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR
            ):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
            Peaks closer than 2 * tolerance are merged before scoring.
        mz_power:
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        noise_cutoff:
            Minimum relative intensity for a peak to be considered. Default is 0.01.
        remove_precursor:
            Whether to remove peaks with m/z values larger than the precursor-m/z (plus offset).
        offset_to_precursor:
            The offset to add to the precursor-m/z when removing peaks.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power
        self.noise_cutoff = noise_cutoff
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor

    def _prepare_spectrum(self, spectrum: SpectrumType) -> np.ndarray:
        """Preprocess and merge one spectrum for linear cosine scoring."""
        peaks = _process_spectrum_peaks(
            spectrum,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
        )
        return sirius_merge_close_peaks(peaks, self.tolerance)

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> tuple[float, int]:
        """Calculate linear cosine score between two spectra.

        Parameters
        ----------
        spectrum_1
            Single reference spectrum.
        spectrum_2
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with cosine score and number of matched peaks.
        """
        spec1 = self._prepare_spectrum(spectrum_1)
        spec2 = self._prepare_spectrum(spectrum_2)

        score, matches = linear_cosine_score(
            spec1,
            spec2,
            self.tolerance,
            self.mz_power,
            self.intensity_power,
        )
        return np.asarray((score, matches), dtype=self.score_datatype)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
    ):
        """Calculate a matrix of linear cosine scores.

        Spectra are preprocessed and merged once before pairwise scoring. This
        avoids repeating precursor removal, noise filtering, and peak merging for
        every spectrum pair.
    
        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If None, compare ``spectra_1`` against
            itself.
        score_fields
            Score fields to return.
        progress_bar
            If True, display a progress bar.
        """
        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)
        selected_fields = self._resolve_score_fields(score_fields)

        n_rows = len(spectra_1)
        n_cols = len(spectra_2)
        result = self._create_dense_result(n_rows, n_cols, selected_fields)

        merged_refs = [
            self._prepare_spectrum(spectrum)
            for spectrum in spectra_1
        ]
        if is_symmetric:
            merged_queries = merged_refs
        else:
            merged_queries = [
                self._prepare_spectrum(spectrum)
                for spectrum in spectra_2
            ]

        for i_ref in tqdm(
            range(n_rows),
            desc="Calculating similarities",
            disable=not progress_bar,
        ):
            if is_symmetric and self.is_commutative:
                query_range = range(i_ref, n_cols)
            else:
                query_range = range(n_cols)

            for i_query in query_range:
                score, matches = linear_cosine_score(
                    merged_refs[i_ref],
                    merged_queries[i_query],
                    self.tolerance,
                    self.mz_power,
                    self.intensity_power,
                )

                score_array = self._as_score((score, matches))

                self._store_in_dense_result(
                    result,
                    i_ref,
                    i_query,
                    score_array,
                    selected_fields,
                )

                if is_symmetric and self.is_commutative and i_ref != i_query:
                    self._store_in_dense_result(
                        result,
                        i_query,
                        i_ref,
                        score_array,
                        selected_fields,
                    )

        return Scores(result)
