import numpy as np
from numba import njit  # type: ignore[attr-defined, import-untyped]


@njit(cache=True)
def sirius_merge_close_peaks(spec, mz_tolerance):
    """Merge close peaks following the Sirius/BOECKER lab algorithm.

    Peaks are merged greedily in descending intensity order. Each unconsumed peak
    adopts its own m/z and sums the intensities of all unconsumed neighbors within
    a merge window of 2 * mz_tolerance. The result is guaranteed to have consecutive
    m/z gaps > 2 * mz_tolerance.

    Parameters
    ----------
    spec
        2D array (N, 2) with columns [mz, intensity], sorted by ascending m/z.
    mz_tolerance
        Tolerance for scoring. Merge window is 2 * mz_tolerance.

    Returns
    -------
    numpy.ndarray
        (M, 2) array of merged peaks sorted by ascending m/z.
    """
    n = spec.shape[0]
    if n == 0:
        return spec.copy()

    merge_window = 2.0 * mz_tolerance

    # Sort indices by ascending intensity and iterate in reverse (descending)
    order = np.argsort(spec[:, 1])

    consumed = np.zeros(n, dtype=np.bool_)
    merged_mz = np.empty(n, dtype=np.float64)
    merged_int = np.empty(n, dtype=np.float64)
    count = 0

    for k in range(n - 1, -1, -1):
        i = order[k]
        if consumed[i]:
            continue

        # This peak becomes the representative; start with its intensity
        mz_i = spec[i, 0]
        total_intensity = spec[i, 1]
        consumed[i] = True

        # Scan left in m/z order
        j = i - 1
        while j >= 0:
            if consumed[j]:
                j -= 1
                continue
            if mz_i - spec[j, 0] > merge_window:
                break
            total_intensity += spec[j, 1]
            consumed[j] = True
            j -= 1

        # Scan right in m/z order
        j = i + 1
        while j < n:
            if consumed[j]:
                j += 1
                continue
            if spec[j, 0] - mz_i > merge_window:
                break
            total_intensity += spec[j, 1]
            consumed[j] = True
            j += 1

        merged_mz[count] = mz_i
        merged_int[count] = total_intensity
        count += 1

    # Build result and sort by ascending m/z
    result = np.empty((count, 2), dtype=np.float64)
    result[:, 0] = merged_mz[:count]
    result[:, 1] = merged_int[:count]
    sort_idx = np.argsort(result[:, 0])
    return result[sort_idx]


@njit(cache=True)
def linear_cosine_score(spec1, spec2, tolerance, mz_power, intensity_power):
    """Compute the LinearCosine similarity between two well-separated spectra.

    Both spectra must have consecutive m/z gaps > 2 * tolerance (as ensured by
    sirius_merge_close_peaks). Uses an O(n+m) two-pointer sweep.

    Parameters
    ----------
    spec1
        2D array (N, 2) with columns [mz, intensity], sorted ascending m/z.
    spec2
        2D array (M, 2) with columns [mz, intensity], sorted ascending m/z.
    tolerance
        Maximum allowed difference between m/z values for a match.
    mz_power
        Power to raise m/z values to.
    intensity_power
        Power to raise intensity values to.

    Returns
    -------
    score : float
        Cosine similarity score.
    matches : int
        Number of matched peak pairs.
    """
    n1 = spec1.shape[0]
    n2 = spec2.shape[0]

    if n1 == 0 or n2 == 0:
        return 0.0, 0

    # Compute weighted products for each spectrum
    products1 = np.empty(n1, dtype=np.float64)
    for i in range(n1):
        products1[i] = (spec1[i, 0] ** mz_power) * (spec1[i, 1] ** intensity_power)

    products2 = np.empty(n2, dtype=np.float64)
    for i in range(n2):
        products2[i] = (spec2[i, 0] ** mz_power) * (spec2[i, 1] ** intensity_power)

    # Compute norms
    norm1 = 0.0
    for i in range(n1):
        norm1 += products1[i] * products1[i]
    norm1 = np.sqrt(norm1)

    norm2 = 0.0
    for i in range(n2):
        norm2 += products2[i] * products2[i]
    norm2 = np.sqrt(norm2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0, 0

    # Two-pointer sweep
    matched_sum = 0.0
    matches = 0
    i = 0
    j = 0
    while i < n1 and j < n2:
        diff = spec1[i, 0] - spec2[j, 0]
        if abs(diff) <= tolerance:
            matched_sum += products1[i] * products2[j]
            matches += 1
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1

    score = matched_sum / (norm1 * norm2)
    return score, matches
