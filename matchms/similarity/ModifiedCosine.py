from typing import Tuple
import numpy
from matchms.similarity.collect_peak_pairs import collect_peak_pairs
from matchms.typing import SpectrumType


class ModifiedCosine:
    """Calculate 'modified cosine score' between mass spectra.

    The modified cosine score aims at quantifying the similarity between two
    mass spectra. The score is calculated by finding best possible matches between
    peaks of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance', or if their m/z ratios
    lie within the tolerance once a mass-shift is applied. The mass shift is
    simply the difference in precursor-m/z between the two spectra.
    See Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743]
    for further details.

    Args:
    -----
    tolerance: float
        Peaks will be considered a match when <= tolerance a part. Default is 0.1.
    """
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> Tuple[float, int]:
        """Calculate modified cosine score between two spectra.
        Args:
        -----
        spectrum1: SpectrumType
            Input spectrum 1.
        spectrum2: SpectrumType
            Input spectrum 2.

        Returns:
        --------

        Tuple with cosine score and number of matched peaks.
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
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0)
            message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            assert spectrum1.get("precursor_mz") and spectrum2.get("precursor_mz"), message
            mass_shift = spectrum1.get("precursor_mz") - spectrum2.get("precursor_mz")
            nonzero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=mass_shift)
            unsorted_matching_pairs = zero_pairs + nonzero_pairs
            return sorted(unsorted_matching_pairs, key=lambda x: x[2], reverse=True)

        def calc_score():
            """Calculate modified cosine score."""
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
