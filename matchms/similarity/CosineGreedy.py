from typing import Tuple
import numpy
from matchms.similarity.collect_peak_pairs import collect_peak_pairs
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
    def __init__(self, tolerance=0.1):
        """
        Args:
        ----
        tolerance: float
            Peaks will be considered a match when <= tolerance appart. Default is 0.1.
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
        """
        def get_peaks_arrays():
            """Get peaks mz and intensities as numpy array."""
            spec1 = numpy.vstack((spectrum1.peaks.mz, spectrum1.peaks.intensities)).T
            spec2 = numpy.vstack((spectrum2.peaks.mz, spectrum2.peaks.intensities)).T
            assert max(spec1[:, 1]) <= 1, ("Input spectrum1 is not normalized. ",
                                           "Apply 'normalize_intensities' filter first.")
            assert max(spec2[:, 1]) <= 1, ("Input spectrum2 is not normalized. ",
                                           "Apply 'normalize_intensities' filter first.")
            return spec1, spec2

        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0)
            matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)
            return matching_pairs

        def calc_score():
            """Calculate cosine similarity score."""
            used1 = set()
            used2 = set()
            score = 0.0
            used_matches = []
            for match in matching_pairs:
                if not match[0] in used1 and not match[1] in used2:
                    score += match[2]
                    used1.add(match[0])  # Every peak can only be paired once
                    used2.add(match[1])  # Every peak can only be paired once
                    used_matches.append(match)
            # Normalize score:
            score = score/max(numpy.sum(spec1[:, 1]**2), numpy.sum(spec2[:, 1]**2))
            return score, len(used_matches)

        spec1, spec2 = get_peaks_arrays()
        matching_pairs = get_matching_pairs()
        return calc_score()
