import logging
import numpy as np
from matchms.typing import SpectrumType
from ._precursor_validation import get_valid_precursor_mz
from .base_similarity import BaseSimilarityWithSparse
from .cosine_greedy import CosineGreedy
from .default_parameters import (
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .spectrum_similarity_functions import _preprocess_peak_array, collect_peak_pairs, score_best_matches


logger = logging.getLogger("matchms")


class ModifiedCosineGreedy(BaseSimilarityWithSparse):
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

    Unlike in matchms < 1.0, this method also applies a noise filter by default,
    which removes peaks with intensity below a certain cutoff. This is typically
    highly beneficial for the performance of the greedy algorithm, and for most
    applications the results are very similar to the exact assignment variant.
    If you want to disable this noise filtering, you can set ``noise_cutoff`` to 0 or None.
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]
    score_fields = ("score", "matches")

    def __init__(
            self, tolerance: float = DEFAULT_MZ_TOLERANCE,
            mz_power: float = 0.0,
            intensity_power: float = DEFAULT_INTENSITY_POWER,
            noise_cutoff: float = DEFAULT_NOISE_CUTOFF,
            remove_precursor: bool = True,
            offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR
            ):
        """Initialize approximate modified cosine.

        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.01.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
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

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> tuple[float, int]:
        """Calculate approximate modified cosine score between two spectra."""

        precursor_mz_1 = get_valid_precursor_mz(spectrum_1, logger)
        precursor_mz_2 = get_valid_precursor_mz(spectrum_2, logger)
        mass_shift = precursor_mz_1 - precursor_mz_2

        if abs(mass_shift) <= self.tolerance:
            return CosineGreedy(
                tolerance=self.tolerance,
                mz_power=self.mz_power,
                intensity_power=self.intensity_power,
            ).pair(spectrum_1, spectrum_2)

        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=0.0,
                mz_power=self.mz_power, intensity_power=self.intensity_power
            )
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

        spec1 = _preprocess_peak_array(
            spectrum_1.peaks.to_numpy,
            precursor_mz=precursor_mz_1,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
        )
        spec2 = _preprocess_peak_array(
            spectrum_2.peaks.to_numpy,
            precursor_mz=precursor_mz_2,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
        )

        matching_pairs = get_matching_pairs()
        if matching_pairs.shape[0] == 0:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2, self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
