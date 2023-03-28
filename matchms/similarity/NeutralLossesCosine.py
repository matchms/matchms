import logging
from typing import Tuple
import numpy as np
from matchms.filtering.add_precursor_mz import _convert_precursor_mz
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .spectrum_similarity_functions import (collect_peak_pairs,
                                            score_best_matches)


logger = logging.getLogger("matchms")


class NeutralLossesCosine(BaseSimilarity):
    """Calculate 'neutral losses cosine score' between mass spectra.

    The neutral losses cosine score aims at quantifying the similarity between two
    mass spectra. The score is calculated by finding best possible matches between
    peaks of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance' once a mass-shift is applied.
    The mass shift is the difference in precursor-m/z between the two spectra.
    In general, `ModifiedCosine` is recommended over `NeutralLossesCosine` because
    it will on average deliver more reliable results.

    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0, ignore_peaks_above_precursor: bool = True):
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
        ignore_peaks_above_precursor:
            By default this is set to True, meaning that peaks with m/z values larger
            than the precursor-m/z will be ignored (since those would correspond to negative
            "neutral losses").
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power
        self.ignore_peaks_above_precursor = ignore_peaks_above_precursor

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
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
        def get_valid_precursor_mz(spectrum):
            """Extract valid precursor_mz from spectrum if possible. If not raise exception."""
            message_precursor_missing = \
                "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            message_precursor_no_number = \
                "Precursor_mz must be of type int or float. Apply 'add_precursor_mz' filter first."
            message_precursor_below_0 = "Expect precursor to be positive number." \
                                        "Apply 'require_precursor_mz' first"

            precursor_mz = spectrum.get("precursor_mz", None)
            if not isinstance(precursor_mz, (int, float)):
                logger.warning(message_precursor_no_number)
            precursor_mz = _convert_precursor_mz(precursor_mz)
            assert precursor_mz is not None, message_precursor_missing
            assert precursor_mz > 0, message_precursor_below_0
            return precursor_mz

        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=mass_shift,
                                                mz_power=self.mz_power,
                                                intensity_power=self.intensity_power)
            if matching_pairs is None:
                return None
            if matching_pairs.shape[0] > 0:
                matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        precursor_mz_ref = get_valid_precursor_mz(reference)
        precursor_mz_query = get_valid_precursor_mz(query)
        mass_shift = precursor_mz_ref - precursor_mz_query

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        if self.ignore_peaks_above_precursor:
            spec1 = spec1[np.where(spec1[:, 0] < precursor_mz_ref)]
            spec2 = spec2[np.where(spec2[:, 0] < precursor_mz_query)]
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2,
                                   self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
