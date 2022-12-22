from typing import List, Tuple
import numba
import numpy as np


@numba.njit
def collect_peak_pairs(spec1: np.ndarray, spec2: np.ndarray,
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
    matches = find_matches(spec1[:, 0], spec2[:, 0], tolerance, shift)
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i, idx in enumerate(idx1):
        power_prod_spec1 = (spec1[idx, 0] ** mz_power) * (spec1[idx, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2[i], 0] ** mz_power) * (spec2[idx2[i], 1] ** intensity_power)
        matching_pairs.append([idx1[i], idx2[i], power_prod_spec1 * power_prod_spec2])
    return np.array(matching_pairs.copy())


@numba.njit
def find_matches(spec1_mz: np.ndarray, spec2_mz: np.ndarray,
                 tolerance: float, shift: float = 0) -> List[Tuple[int, int]]:
    """Faster search for matching peaks.
    Makes use of the fact that spec1 and spec2 contain ordered peak m/z (from
    low to high m/z).

    Parameters
    ----------
    spec1_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    spec2_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
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
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches


@numba.njit(fastmath=True)
def score_best_matches(matching_pairs: np.ndarray, spec1: np.ndarray,
                       spec2: np.ndarray, mz_power: float = 0.0,
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

    score = score/(np.sum(spec1_power ** 2) ** 0.5 * np.sum(spec2_power ** 2) ** 0.5)
    return score, used_matches


@numba.njit
def number_matching(numbers_1, numbers_2, tolerance):
    """Find all pairs between numbers_1 and numbers_2 which are within tolerance.
    """
    rows = []
    cols = []
    data = []
    for i, number_1 in enumerate(numbers_1):
        for j, number_2 in enumerate(numbers_2):
            value = (abs(number_1 - number_2) <= tolerance)
            if value:
                data.append(value)
                rows.append(i)
                cols.append(j)
    return np.array(rows), np.array(cols), np.array(data)


@numba.njit
def number_matching_symmetric(numbers_1, tolerance):
    """Find all pairs between numbers_1 and numbers_1 which are within tolerance.
    """
    rows = []
    cols = []
    data = []
    for i, number_1 in enumerate(numbers_1):
        for j in range(i, len(numbers_1)):
            value = (abs(number_1 - numbers_1[j]) <= tolerance)
            if value:
                data.append(value)
                rows.append(i)
                cols.append(j)
                if i != j:
                    data.append(value)
                    rows.append(j)
                    cols.append(i)
    return np.array(rows), np.array(cols), np.array(data)


@numba.njit
def number_matching_ppm(numbers_1, numbers_2, tolerance_ppm):
    """Find all pairs between numbers_1 and numbers_2 which are within tolerance.
    """
    rows = []
    cols = []
    data = []
    for i, number_1 in enumerate(numbers_1):
        for j, number_2 in enumerate(numbers_2):
            mean_value = (number_1 + number_2)/2
            value = (abs(number_1 - number_2)/mean_value * 1e6 <= tolerance_ppm)
            if value:
                data.append(value)
                rows.append(i)
                cols.append(j)
    return np.array(rows), np.array(cols), np.array(data)


@numba.njit
def number_matching_symmetric_ppm(numbers_1, tolerance_ppm):
    """Find all pairs between numbers_1 and numbers_1 which are within tolerance.
    """
    rows = []
    cols = []
    data = []
    for i, number_1 in enumerate(numbers_1):
        for j in range(i, len(numbers_1)):
            mean_value = (number_1 + numbers_1[j])/2
            value = (abs(number_1 - numbers_1[j])/mean_value * 1e6 <= tolerance_ppm)
            if value:
                data.append(value)
                rows.append(i)
                cols.append(j)
                if i != j:
                    data.append(value)
                    rows.append(j)
                    cols.append(i)
    return np.array(rows), np.array(cols), np.array(data)
