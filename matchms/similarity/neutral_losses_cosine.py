import logging
import numpy as np
from matchms.typing import SpectrumType
from ._precursor_validation import get_valid_precursor_mz
from .base_similarity import BaseSimilarityWithSparse
from .default_parameters import (
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .spectrum_similarity_functions import _preprocess_peak_array, collect_peak_pairs, score_best_matches


logger = logging.getLogger("matchms")

class NeutralLossesCosine(BaseSimilarityWithSparse):
    """Calculate 'neutral losses cosine score' between mass spectra.

    The neutral losses cosine score aims at quantifying the similarity between two
    mass spectra. The score is calculated by finding best possible matches between
    peaks of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance' once a mass-shift is applied.
    The mass shift is the difference in precursor-m/z between the two spectra.
    In general, `ModifiedCosineGreedy` is recommended over `NeutralLossesCosine` because
    it will on average deliver more reliable results.

    """

    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]
    score_fields = ("score", "matches")

    def __init__(
            self,
            tolerance: float = DEFAULT_MZ_TOLERANCE,
            mz_power: float = DEFAULT_MZ_POWER,
            intensity_power: float = DEFAULT_INTENSITY_POWER,
            noise_cutoff: float = DEFAULT_NOISE_CUTOFF,
            remove_precursor: bool = True,
            offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR,
            ):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        noise_cutoff:
            Minimum relative intensity for a peak to be considered. Default is 0.01.
        remove_precursor:
            If True and ``precursor_mz`` metadata are available, remove peaks above
            ``precursor_mz + offset_to_precursor`` before scoring.
        offset_to_precursor:
            Offset used when ``remove_precursor=True``. This will only keep
            m/z values <= precursor_mz + offset_to_precursor. Default is -1.6 Da.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power
        self.noise_cutoff = noise_cutoff
        self.remove_precursor = remove_precursor
        self.offset_to_precursor = offset_to_precursor

    def _get_matching_pairs(self, spec1, spec2, mass_shift: float) -> np.ndarray:
        """Find all pairs of peaks that match within the given tolerance."""
        matching_pairs = collect_peak_pairs(
            spec1, spec2, self.tolerance,
            shift=mass_shift, mz_power=self.mz_power,
            intensity_power=self.intensity_power
        )
        if matching_pairs is None:
            return None
        if matching_pairs.shape[0] > 0:
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1], :]
        return matching_pairs
    
    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> tuple[float, int]:
        """Calculate neutral losses cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------

        Tuple with cosine score and number of matched peaks.
        """

        precursor_mz_1 = get_valid_precursor_mz(spectrum_1, logger)
        precursor_mz_2 = get_valid_precursor_mz(spectrum_2, logger)
        mass_shift = precursor_mz_1 - precursor_mz_2

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

        matching_pairs = self._get_matching_pairs(spec1, spec2, mass_shift)
        if matching_pairs is None:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2, self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
