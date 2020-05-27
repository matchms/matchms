from typing import Tuple
import numba
import numpy
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
    ----
    tolerance: float
        Peaks will be considered a match when <= tolerance appart. Default is 0.1.
    """
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> Tuple[float, int]:
        """Calculate modified cosine score between two spectra.
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
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = find_pairs_numba(spec1, spec2, self.tolerance, shift=0.0)
            message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            assert spectrum1.get("precursor_mz") and spectrum2.get("precursor_mz"), message
            mass_shift = spectrum1.get("precursor_mz") - spectrum2.get("precursor_mz")
            nonzero_pairs = find_pairs_numba(spec1, spec2, self.tolerance, shift=mass_shift)
            matching_pairs = zero_pairs + nonzero_pairs
            matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)
            return matching_pairs

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


@numba.njit
def find_pairs_numba(spec1, spec2, tolerance, shift=0):
    """Find matching pairs between two spectra.

    Args
    ----
    spec1: numpy array
        Spectrum peaks and intensities as numpy array.
    spec2: numpy array
        Spectrum peaks and intensities as numpy array.
    tolerance : float
        Peaks will be considered a match when <= tolerance appart.
    shift : float, optional
        Shift spectra peaks by shift. The default is 0.

    Returns
    -------
    matching_pairs : list
        List of found matching peaks.
    """
    matching_pairs = []

    for idx in range(len(spec1)):
        intensity = spec1[idx, 1]
        matches = numpy.where((numpy.abs(spec2[:, 0] - spec1[idx, 0] + shift) <= tolerance))[0]
        for match in matches:
            matching_pairs.append((idx, match, intensity*spec2[match][1]))

    return matching_pairs
