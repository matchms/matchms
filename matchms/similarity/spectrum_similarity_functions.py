from typing import Tuple
import numba
import numpy
from matchms.typing import SpectrumType


@numba.njit
def collect_peak_pairs(spec1, spec2, tolerance, shift=0):
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


    def get_peaks_array(spectrum: SpectrumType) -> numpy.ndarray:
        """Get peaks mz and intensities as numpy array."""
        peaks_array = numpy.vstack((spectrum.peaks.mz, spectrum.peaks.intensities)).T
        assert max(peaks_array[:, 1]) <= 1, ("Input spectrum is not normalized. ",
                                             "Apply 'normalize_intensities' filter first.")
        return peaks_array


    def score_best_matches(matching_pairs: list) -> Tuple[float, int]:
        """Calculate cosine-like score by multiplying matches. Does recuire a sorted
        list of matching peaks (sorted by intensity product)."""
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
