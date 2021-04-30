from typing import Tuple
import numpy
from scipy.optimize import linear_sum_assignment
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class CosineHungarian(BaseSimilarity):
    """Calculate 'cosine similarity score' between two spectra (using Hungarian algorithm).

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here solved using the Hungarian algorithm.
    This can perform notably slower than the 'greedy' implementation in
    :class:`~matchms.similarity.CosineGreedy`, but does represent a mathematically proper
    solution to the problem.
    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", numpy.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns:
        --------

        Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0,
                                                mz_power=self.mz_power,
                                                intensity_power=self.intensity_power)
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[numpy.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        def get_matching_pairs_matrix():
            """Create matrix of multiplied intensities of all matching pairs
            between spectrum1 and spectrum2.
            Returns
            paired_peaks1:
                list of paired peaks of spectrum1
            paired_peaks2:
                list of paired peaks of spectrum2
            matching_pairs_matrix:
                Array of multiplied intensities between all matching peaks.
            """
            if matching_pairs is None:
                return None, None, None
            paired_peaks1 = list(set(matching_pairs[:, 0]))
            paired_peaks2 = list(set(matching_pairs[:, 1]))
            matrix_size = (len(paired_peaks1), len(paired_peaks2))
            matching_pairs_matrix = numpy.ones(matrix_size)
            for i in range(matching_pairs.shape[0]):
                matching_pairs_matrix[paired_peaks1.index(matching_pairs[i, 0]),
                                      paired_peaks2.index(matching_pairs[i, 1])] = 1 - matching_pairs[i, 2]
            return paired_peaks1, paired_peaks2, matching_pairs_matrix

        def solve_hungarian():
            """Use hungarian algorithm to solve the linear sum assignment problem."""
            row_ind, col_ind = linear_sum_assignment(matching_pairs_matrix)
            score = len(row_ind) - matching_pairs_matrix[row_ind, col_ind].sum()
            used_matches = [(paired_peaks1[x], paired_peaks2[y]) for (x, y) in zip(row_ind, col_ind)]
            return score, used_matches

        def calc_score():
            """Calculate cosine similarity score."""
            if matching_pairs_matrix is None:
                return numpy.asarray((0.0, 0), dtype=self.score_datatype)
            score, used_matches = solve_hungarian()
            # Normalize score:
            spec1_power = numpy.power(spec1[:, 0], self.mz_power) \
                * numpy.power(spec1[:, 1], self.intensity_power)
            spec2_power = numpy.power(spec2[:, 0], self.mz_power) \
                * numpy.power(spec2[:, 1], self.intensity_power)
            score = score/(numpy.sqrt(numpy.sum(spec1_power**2)) * numpy.sqrt(numpy.sum(spec2_power**2)))
            return numpy.asarray((score, len(used_matches)), dtype=self.score_datatype)

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        paired_peaks1, paired_peaks2, matching_pairs_matrix = get_matching_pairs_matrix()
        return calc_score()
