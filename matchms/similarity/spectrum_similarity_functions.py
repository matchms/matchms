from typing import Tuple
import numba
import numpy
from matchms.typing import SpectrumType


@numba.njit
def collect_peak_pairs(spec1: numpy.ndarray, spec2: numpy.ndarray,
                       tolerance: float, shift: float = 0, mz_power: float = 0.0,
                       intensity_power: float = 1.0):
    # pylint: disable=too-many-arguments
    """Find matching pairs between two spectra.

    Args
    ----
    spec1:
        Spectrum peaks and intensities as numpy array.
    spec2:
        Spectrum peaks and intensities as numpy array.
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    shift
        Shift spectra peaks by shift. The default is 0.
    mz_power:
        The power to raise mz to in the cosine function. The default is 0, in which
        case the peak intensity products will not depend on the m/z ratios.
    intensity_power:
        The power to raise intensity to in the cosine function. The default is 1.

    Returns
    -------
    matching_pairs : numpy array
        Array of found matching peaks.
    """
    matching_pairs = []

    for idx in range(spec1.shape[0]):
        intensity = spec1[idx, 1]
        mz = spec1[idx, 0]
        matches = numpy.where((numpy.abs(spec2[:, 0] - spec1[idx, 0] + shift) <= tolerance))[0]
        for match in matches:
            power_prod_spec1 = ((mz ** mz_power) * (intensity ** intensity_power))
            power_prod_spec2 = ((spec2[match][0] ** mz_power) * (spec2[match][1] ** intensity_power))
            matching_pairs.append((idx, match, power_prod_spec1 * power_prod_spec2))

    if len(matching_pairs) > 0:
        return numpy.array(matching_pairs)
    return numpy.empty((0, 0))


def get_peaks_array(spectrum: SpectrumType) -> numpy.ndarray:
    """Get peaks mz and intensities as numpy array."""
    return numpy.vstack((spectrum.peaks.mz, spectrum.peaks.intensities)).T


@numba.njit(fastmath=True)
def score_best_matches(matching_pairs: numpy.ndarray, spec1: numpy.ndarray,
                       spec2: numpy.ndarray, mz_power: float = 0.0,
                       intensity_power: float = 1.0) -> Tuple[float, int]:
    """Calculate cosine-like score by multiplying matches. Does require a sorted
    list of matching peaks (sorted by intensity product)."""
    if matching_pairs.shape[0] == 0:
        return 0.0, 0
    used1 = set()
    used2 = set()
    score = 0.0
    used_matches = 0
    for i in range(matching_pairs.shape[0]):
        if not matching_pairs[i, 0] in used1 and not matching_pairs[i, 1] in used2:
            score += matching_pairs[i, 2]
            used1.add(matching_pairs[i, 0])  # Every peak can only be paired once
            used2.add(matching_pairs[i, 1])  # Every peak can only be paired once
            used_matches += 1

    # Normalize score:
    spec1_power = spec1[:, 0] ** mz_power * spec1[:, 1] ** intensity_power
    spec2_power = spec2[:, 0] ** mz_power * spec2[:, 1] ** intensity_power

    score = score/(numpy.sum(spec1_power ** 2) ** 0.5 * numpy.sum(spec2_power ** 2) ** 0.5)
    return score, used_matches
