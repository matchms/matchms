from typing import Tuple
import numpy as np
from numba import njit
from numba.typed import List


@njit
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
    matches: List = find_matches(spec1[:, 0], spec2[:, 0], tolerance, shift)
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i, idx in enumerate(idx1):
        power_prod_spec1 = (spec1[idx, 0] ** mz_power) * (spec1[idx, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2[i], 0] ** mz_power) * (spec2[idx2[i], 1] ** intensity_power)
        matching_pairs.append([idx, idx2[i], power_prod_spec1 * power_prod_spec2])
    return np.array(matching_pairs.copy())


@njit
def find_matches(spec1_mz: np.ndarray, spec2_mz: np.ndarray,
                 tolerance: float, shift: float = 0) -> List:
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
    matches = List()
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx + 1
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches


@njit
def find_matches_ppm(
    spec1_mz: np.ndarray,
    spec2_mz: np.ndarray,
    tolerance_ppm: float,
    shift: float = 0
    ) -> List:
    """Faster search for matching peaks.
    Makes use of the fact that spec1 and spec2 contain ordered peak m/z (from
    low to high m/z).

    Parameters
    ----------
    spec1_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    spec2_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    tolerance_ppm
        Peaks will be considered a match within ppm based tolerance.
    shift
        Shift peaks of second spectra by shift. The default is 0.

    Returns
    -------
    matches
        List containing entries of type (idx1, idx2).

    """
    lowest_idx = 0
    matches = List()
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        tolerance = mz * 1e-6 * tolerance_ppm
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx + 1
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches


@njit(fastmath=True)
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


@njit
def collect_peak_pairs_entropy(
    spec1: np.ndarray, spec2: np.ndarray,
    tolerance: float, use_ppm: bool = False, shift: float = 0.0,
) -> np.ndarray:
    """
    Find matching peak pairs and compute per-pair entropy contributions.

    Parameters
    ----------
    spec1 : np.ndarray, shape=(n,2)
        Spectrum A as rows [mz, intensity].
    spec2 : np.ndarray, shape=(m,2)
        Spectrum B as rows [mz, intensity].
    tolerance : float
        Tolerance in Da or ppm (if use_ppm).
    use_ppm : bool
        If True, interpret tolerance in ppm.
    shift : float
        Global m/z shift for spec2.

    Returns
    -------
    np.ndarray of shape (k,3)
        Each row [idx1, idx2, entropy_contribution]. Sorted by idx1.
        Returns None if no matches.
    """
    # Choose matching index pairs
    if use_ppm:
        matches = find_matches_ppm(spec1[:,0], spec2[:,0], tolerance, shift)
    else:
        matches = find_matches(spec1[:,0], spec2[:,0], tolerance, shift)
    if len(matches) == 0:
        return None

    """ possible alternative (but seems less efficient)
    # Compute normalization divisors
    if total_norm:
        div1 = spec1[:,1].sum()
        div2 = spec2[:,1].sum()
    else:
        div1 = spec1[:,1].max()
        div2 = spec2[:,1].max()

    # Allocate output
    out = List()
    for pair in matches:
        i, j = pair
        p1 = spec1[i,1] / div1 if div1 > 0 else 0.0
        p2 = spec2[j,1] / div2 if div2 > 0 else 0.0
        # entropy contribution: p1*log(2*p1/(p1+p2)) + p2*log(2*p2/(p1+p2))
        contrib = 0.0
        if p1 > 0:
            contrib += p1 * np.log(2 * p1 / (p1 + p2))
        if p2 > 0:
            contrib += p2 * np.log(2 * p2 / (p1 + p2))
        out.append((i, j, contrib))

    # Convert to numpy array
    arr = np.empty((len(out),3), dtype=np.float64)
    for idx in range(len(out)):
        arr[idx,0] = out[idx][0]
        arr[idx,1] = out[idx][1]
        arr[idx,2] = out[idx][2]
    return arr
    """
    idx1 = [x[0] for x in matches]
    idx2 = [x[1] for x in matches]
    if len(idx1) == 0:
        return None
    matching_pairs = []
    for i, idx in enumerate(idx1):
        prod_spec = spec1[idx, 1] * spec2[idx2[i], 1]  # use product of intensities
        matching_pairs.append([idx, idx2[i], prod_spec])
    return np.array(matching_pairs.copy())


@njit(fastmath=True)
def score_best_matches_entropy(
    matching_pairs: np.ndarray,
    spec1: np.ndarray,
    spec2: np.ndarray,
    total_norm: bool = True
) -> Tuple[float, int]:
    """
    Greedy selection of peak pairs maximizing entropy contributions,
    plus entropy from unmatched peaks.

    Parameters
    ----------
    matching_pairs : np.ndarray, shape=(k,3)
        Rows [idx1, idx2, contrib] sorted descending by contrib.
    spec1, spec2 : np.ndarray, shape=(n,2) and (m,2)
        Spectra as [mz, intensity].
    total_norm : bool
        Use sum-normalization (True) or max-normalization (False).

    Returns
    -------
    (score, n_matches)
        score : total entropy similarity score (unnormalized loss)
        n_matches : number of matched peak pairs used
    """
    score = 0.0
    used1 = set()
    used2 = set()

    # Compute normalization
    if total_norm:
        div1 = spec1[:,1].sum()
        div2 = spec2[:,1].sum()
    else:
        div1 = spec1[:,1].max()
        div2 = spec2[:,1].max()

    # Sum contributions from unique matches
    for row in matching_pairs:
        i = int(row[0])
        j = int(row[1])
        if (i not in used1) and (j not in used2):
            p1 = spec1[i,1] / div1 if div1 > 0 else 0.0
            p2 = spec2[j,1] / div2 if div2 > 0 else 0.0
            # entropy contribution: p1*log(2*p1/(p1+p2)) + p2*log(2*p2/(p1+p2))
            if p1 > 0:
                score += p1 * np.log(2 * p1 / (p1 + p2))
            if p2 > 0:
                score += p2 * np.log(2 * p2 / (p1 + p2))
            used1.add(i)
            used2.add(j)

    # Add unmatched peak entropy: p*log(2)
    # Unmatched in spec1
    for i in range(spec1.shape[0]):
        if i not in used1:
            p = spec1[i,1] / div1 if div1 > 0 else 0.0
            if p > 0:
                score += p * np.log(2)
    # Unmatched in spec2
    for j in range(spec2.shape[0]):
        if j not in used2:
            p = spec2[j,1] / div2 if div2 > 0 else 0.0
            if p > 0:
                score += p * np.log(2)

    return score / np.log(4.0)


@njit
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


@njit
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


@njit
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


@njit
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
