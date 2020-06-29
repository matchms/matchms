from typing import Tuple
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.similarity.spectrum_similarity_functions import get_peaks_array
from matchms.similarity.spectrum_similarity_functions import score_best_matches
from matchms.typing import SpectrumType


class CosineGreedy:
    """Calculate 'cosine similarity score' between two spectra.

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here solved in a 'greedy' way.
    This can perform notably faster, but does occasionally deviate slightly from
    a fully correct solution (as with the Hungarian algorithm). In practice this
    will rarely affect similarity scores notably, in particular for smaller
    tolerances.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import CosineGreedy

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]))
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]))

        # Use factory to construct a similarity function
        cosine_greedy = CosineGreedy(tolerance=0.2)

        score, n_matches = cosine_greedy(spectrum_1, spectrum_2)

        print(f"Cosine score is {score:.2f} with {n_matches} matched peaks")

    Should output

    .. testoutput::

        Cosine score is 0.52 with 1 matched peaks

    """
    def __init__(self, tolerance: float = 0.1):
        """
        Parameters
        ----------
        tolerance
            Peaks will be considered a match when <= tolerance apart.
        """
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> Tuple[float, int]:
        """Calculate cosine score between two spectra.

        Args:
        ----
        spectrum1: SpectrumType
            Input spectrum 1.
        spectrum2: SpectrumType
            Input spectrum 2.

        Returns:
        --------

        Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0)
            matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)
            return matching_pairs

        spec1 = get_peaks_array(spectrum1)
        spec2 = get_peaks_array(spectrum2)
        matching_pairs = get_matching_pairs()
        return score_best_matches(matching_pairs, spec1, spec2)
