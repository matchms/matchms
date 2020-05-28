from typing import Tuple
import numpy
from scipy.optimize import linear_sum_assignment
from matchms.similarity.collect_peak_pairs import collect_peak_pairs
from matchms.typing import SpectrumType


class CosineHungarian:
    """Calculate 'cosine similarity score' between two spectra (using Hungarian algorithm).

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here sovled using the Hungarian algorithm.
    This can perform notably slower than the 'greedy' implementation in CosineGreedy, but
    does represent a mathematically proper solution to the problem.

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
            used_matches = []
            list1 = list({x[0] for x in matching_pairs})
            list2 = list({x[1] for x in matching_pairs})
            matrix_size = (len(list1), len(list2))
            matrix = numpy.ones(matrix_size)

            if len(matching_pairs) > 0:
                for match in matching_pairs:
                    matrix[list1.index(match[0]), list2.index(match[1])] = 1 - match[2]
                # Use hungarian agorithm to solve the linear sum assignment problem
                row_ind, col_ind = linear_sum_assignment(matrix)
                score = len(row_ind) - matrix[row_ind, col_ind].sum()
                used_matches = [(list1[x], list2[y]) for (x, y) in zip(row_ind, col_ind)]
                # Normalize score:
                score = score/max(numpy.sum(spec1[:, 1]**2), numpy.sum(spec2[:, 1]**2))
            else:
                score = 0.0
            return score, len(used_matches)

        spec1, spec2 = get_peaks_arrays()
        matching_pairs = get_matching_pairs()
        return calc_score()
