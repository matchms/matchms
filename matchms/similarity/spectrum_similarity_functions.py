from typing import List
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
    matches = find_matches(spec1, spec2, tolerance, shift)
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i, idx in enumerate(idx1):
        power_prod_spec1 = (spec1[idx, 0] ** mz_power) * (spec1[idx, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2[i], 0] ** mz_power) * (spec2[idx2[i], 1] ** intensity_power)
        matching_pairs.append([idx1[i], idx2[i], power_prod_spec1 * power_prod_spec2])
    return numpy.array(matching_pairs.copy())


@numba.njit
def find_matches(spec1: numpy.ndarray, spec2: numpy.ndarray,
                 tolerance: float, shift: float = 0) -> List[Tuple[int, int]]:
    """Faster search for matching peaks.
    Makes use of the fact that spec1 and spec2 contain ordered peak m/z (from
    low to high m/z).

    Parameters
    ----------
    spec1:
        Spectrum peaks and intensities as numpy array. Peak mz values must be ordered.
    spec2:
        Spectrum peaks and intensities as numpy array. Peak mz values must be ordered.
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    shift
        Shift peaks of second spectra by shift. The default is 0.

    Returns
    -------
    matches
        List containing entries of type (idx1, idx2).

    """
    lowest_idx = 0
    matches = []
    for peak1_idx in range(spec1.shape[0]):
        mz = spec1[peak1_idx, 0]
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2.shape[0]):
            mz2 = spec2[peak2_idx, 0] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches


def get_peaks_array(spectrum: SpectrumType) -> numpy.ndarray:
    """Get peaks mz and intensities as numpy array."""
    return numpy.vstack((spectrum.peaks.mz, spectrum.peaks.intensities)).T


@numba.njit(fastmath=True)
def score_best_matches(matching_pairs: numpy.ndarray, spec1: numpy.ndarray,
                       spec2: numpy.ndarray, mz_power: float = 0.0,
                       intensity_power: float = 1.0) -> Tuple[float, int]:
    """Calculate cosine-like score by multiplying matches. Does require a sorted
    list of matching peaks (sorted by intensity product)."""
    score = float(0.0)
    used_matches = int(0)
    used1 = set()
    used2 = set()
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
