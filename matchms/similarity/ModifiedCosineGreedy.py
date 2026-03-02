import logging
from typing import Tuple
import numpy as np
from matchms.typing import SpectrumType
from ._precursor_validation import get_valid_precursor_mz
from .BaseSimilarity import BaseSimilarity
from .spectrum_similarity_functions import collect_peak_pairs, score_best_matches


logger = logging.getLogger("matchms")


class ModifiedCosineGreedy(BaseSimilarity):
    """Calculate an approximate modified cosine score between mass spectra.

    This implementation solves the peak assignment in a greedy way and is therefore
    an approximation. See :class:`~matchms.similarity.ModifiedCosineHungarian` for
    the exact assignment variant.

    The modified cosine score aims at quantifying the similarity between two
    mass spectra. Two peaks are considered a potential match if their m/z ratios
    lie within the given ``tolerance``, or if their m/z ratios lie within the
    tolerance once a mass-shift is applied. The mass shift is the difference in
    precursor m/z between the two spectra.

    See Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743]
    for further details.
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0, intensity_power: float = 1.0):
        """Initialize approximate modified cosine.

        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate approximate modified cosine score between two spectra."""

        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=0.0,
                mz_power=self.mz_power, intensity_power=self.intensity_power
            )
            precursor_mz_ref = get_valid_precursor_mz(reference, logger)
            precursor_mz_query = get_valid_precursor_mz(query, logger)

            mass_shift = precursor_mz_ref - precursor_mz_query
            nonzero_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=mass_shift,
                mz_power=self.mz_power, intensity_power=self.intensity_power
            )

            if zero_pairs is None:
                zero_pairs = np.zeros((0, 3))
            if nonzero_pairs is None:
                nonzero_pairs = np.zeros((0, 3))
            matching_pairs = np.concatenate((zero_pairs, nonzero_pairs), axis=0)
            if matching_pairs.shape[0] > 0:
                matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        if matching_pairs.shape[0] == 0:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2, self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
