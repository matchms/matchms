import numba
import numpy
from matchms.typing import SpectrumType


class CosineGreedyNumba:
    """Calculate cosine score between two spectra.

    This score is calculated by summing intensiy products of matching peaks between
    two spectra (matching within set tolerance).
    of two spectra.

    Args:
    ----
    tolerance: float
        Peaks will be considered a match when <= tolerance appart. Default is 0.1.
    """
    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> float:
        """Calculate cosine score between two spectra.
        Args:
        ----
        spectrum1: SpectrumType
            Input spectrum 1.
        spectrum2: SpectrumType
            Input spectrum 2.
        """
        def get_normalized_peaks_arrays():
            # Get peaks mz and intensities
            spec1 = numpy.vstack((spectrum1.peaks.mz, spectrum1.peaks.intensities)).T
            spec2 = numpy.vstack((spectrum2.peaks.mz, spectrum2.peaks.intensities)).T
            # Normalize intensities
            spec1[:, 1] = spec1[:, 1]/max(spec1[:, 1])
            spec2[:, 1] = spec2[:, 1]/max(spec2[:, 1])
            return spec1, spec2

        def get_matching_pairs():
            matching_pairs = find_pairs_numba(spec1, spec2, self.tolerance, shift=0.0)
            matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)
            return matching_pairs

        def calc_score():
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

        spec1, spec2 = get_normalized_peaks_arrays()
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
