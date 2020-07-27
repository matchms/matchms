from typing import Tuple
import numba
import numpy
from matchms.typing import SpectrumType


@numba.njit
def collect_peak_pairs(spec1, spec2, tolerance, shift=0,
                       mz_power=0.0, intensity_power=1.0):
    # pylint: disable=too-many-arguments
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
    mz_power: float, optional
        The power to raise mz to in the cosine function. The default is 0, in which
        case the peak intensity products will not depend on the m/z ratios.
    intensity_power: float, optional
        The power to raise intensity to in the cosine function. The default is 1.

    Returns
    -------
    matching_pairs : list
        List of found matching peaks.
    """
    matching_pairs = []

    for idx in range(len(spec1)):
        intensity = spec1[idx, 1]
        mz = spec1[idx, 0]
        matches = numpy.where((numpy.abs(spec2[:, 0] - spec1[idx, 0] + shift) <= tolerance))[0]
        for match in matches:
            power_prod_spec1 = ((mz ** mz_power) * (intensity ** intensity_power))
            power_prod_spec2 = ((spec2[match][0] ** mz_power) * (spec2[match][1] ** intensity_power))
            matching_pairs.append((idx, match, power_prod_spec1 * power_prod_spec2))

    return matching_pairs


def get_peaks_array(spectrum: SpectrumType) -> numpy.ndarray:
    """Get peaks mz and intensities as numpy array."""
    return numpy.vstack((spectrum.peaks.mz, spectrum.peaks.intensities)).T


def score_best_matches(matching_pairs: list, spec1: numpy.ndarray,
                       spec2: numpy.ndarray, mz_power: float = 0.0,
                       intensity_power: float = 1.0) -> Tuple[float, int]:
    """Calculate cosine-like score by multiplying matches. Does require a sorted
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
    spec1_power = numpy.power(spec1[:, 0], mz_power) * numpy.power(spec1[:, 1], intensity_power)
    spec2_power = numpy.power(spec2[:, 0], mz_power) * numpy.power(spec2[:, 1], intensity_power)

    score = score/(numpy.sqrt(numpy.sum(spec1_power**2)) * numpy.sqrt(numpy.sum(spec2_power**2)))
    return score, len(used_matches)
