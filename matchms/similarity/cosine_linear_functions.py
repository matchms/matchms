import numpy as np
from numba import njit  # type: ignore[attr-defined, import-untyped]


@njit(cache=True)
def _sorted_peak_order(spec):
    """Return intensity-sorted indices with deterministic tie-breaking.

    Peaks are primarily sorted by ascending intensity. Equal-intensity peaks are
    reordered by descending original index, so reverse iteration visits the
    lower-m/z peak first because input spectra are already sorted by m/z.
    """
    order = np.argsort(spec[:, 1])
    start = 0
    n = order.shape[0]

    while start < n:
        end = start + 1
        intensity = spec[order[start], 1]
        while end < n and spec[order[end], 1] == intensity:
            end += 1

        for left in range(start, end):
            max_pos = left
            for right in range(left + 1, end):
                if order[right] > order[max_pos]:
                    max_pos = right
            if max_pos != left:
                tmp = order[left]
                order[left] = order[max_pos]
                order[max_pos] = tmp

        start = end

    return order


@njit(cache=True)
def sirius_merge_close_peaks(spec, mz_tolerance):
    """Merge close peaks following the Sirius/BOECKER lab algorithm.

    Peaks are merged greedily in descending intensity order. Each unconsumed peak
    adopts its own m/z and sums the intensities of all unconsumed neighbors within
    a merge window of 2 * mz_tolerance. The result is guaranteed to have consecutive
    m/z gaps > 2 * mz_tolerance. When multiple peaks share the same intensity,
    the lower-m/z peak is processed first so the representative peak stays
    deterministic across NumPy and Numba sort implementations.

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

    # Sort indices by ascending intensity and iterate in reverse (descending).
    order = _sorted_peak_order(spec)

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
def _weighted_products(spec, mz_power, intensity_power):
    """Return per-peak products mz**mz_power * intensity**intensity_power and their L2 norm."""
    n = spec.shape[0]
    products = np.empty(n, dtype=np.float64)
    norm = 0.0
    for i in range(n):
        products[i] = (spec[i, 0] ** mz_power) * (spec[i, 1] ** intensity_power)
        norm += products[i] * products[i]
    return products, np.sqrt(norm)


@njit(cache=True)
def linear_cosine_score(spec1, spec2, tolerance, mz_power, intensity_power):
    """Compute the CosineLinear similarity between two well-separated spectra.

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

    products1, norm1 = _weighted_products(spec1, mz_power, intensity_power)
    products2, norm2 = _weighted_products(spec2, mz_power, intensity_power)

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


@njit(cache=True)
def _is_well_separated(mz, tolerance):
    min_gap = 2.0 * tolerance
    for i in range(1, mz.shape[0]):
        if mz[i] - mz[i - 1] <= min_gap:
            return False
    return True


@njit(cache=True)
def _linear_partners(values1, values2, tolerance, partner1, partner2):
    """Fill mutual partner indices of within-tolerance pairs, -1 where unmatched."""
    n1 = values1.shape[0]
    n2 = values2.shape[0]
    partner1[:] = -1
    partner2[:] = -1
    j = 0
    for i in range(n1):
        while j < n2 and values2[j] < values1[i] - tolerance:
            j += 1
        if j < n2 and values2[j] <= values1[i] + tolerance:
            partner1[i] = j
            partner2[j] = i
            j += 1


@njit(cache=True)
def _walk_path(start, on_left, direct1, shifted1, direct2, shifted2, visited1, visited2, path_left, path_right):
    """Walk the alternating direct/shifted path from an endpoint, returning its edge count."""
    n_edges = 0
    node = start
    kind = -1  # edge kind used to reach node, 0 direct, 1 shifted
    while True:
        nxt = -1
        if on_left:
            visited1[node] = True
            if kind != 0 and direct1[node] >= 0:
                kind = 0
                nxt = direct1[node]
            elif kind != 1 and shifted1[node] >= 0:
                kind = 1
                nxt = shifted1[node]
            if nxt < 0 or visited2[nxt]:
                return n_edges
            path_left[n_edges] = node
            path_right[n_edges] = nxt
        else:
            visited2[node] = True
            if kind != 0 and direct2[node] >= 0:
                kind = 0
                nxt = direct2[node]
            elif kind != 1 and shifted2[node] >= 0:
                kind = 1
                nxt = shifted2[node]
            if nxt < 0 or visited1[nxt]:
                return n_edges
            path_left[n_edges] = nxt
            path_right[n_edges] = node
        n_edges += 1
        node = nxt
        on_left = not on_left


@njit(cache=True)
def _select_path_edges(n_edges, benefits, dp, selected):
    """Mark the maximum-weight set of pairwise non-adjacent path edges."""
    dp[0] = 0.0
    dp[1] = benefits[0]
    for k in range(2, n_edges + 1):
        take = dp[k - 2] + benefits[k - 1]
        skip = dp[k - 1]
        dp[k] = max(take, skip)

    for k in range(n_edges):
        selected[k] = False
    k = n_edges
    while k > 0:
        if k == 1:
            selected[0] = True
            break
        if dp[k - 2] + benefits[k - 1] >= dp[k - 1]:
            selected[k - 1] = True
            k -= 2
        else:
            k -= 1


@njit(cache=True)
def modified_linear_cosine_score(spec1, spec2, precursor_mz1, precursor_mz2, tolerance, mz_power, intensity_power):
    """Compute the exact modified cosine between two well-separated spectra in O(n+m).

    SIRIUS linear-time modified cosine. Direct and precursor-shifted matches are found
    with one two-pointer sweep each, and since every peak has at most one partner of
    each kind their conflicts form paths solved by dynamic programming.

    Parameters
    ----------
    spec1
        2D array (N, 2) with columns [mz, intensity], sorted ascending m/z,
        consecutive m/z gaps > 2 * tolerance (as ensured by sirius_merge_close_peaks).
    spec2
        2D array (M, 2) with the same layout and precondition.
    precursor_mz1
        Precursor m/z of spec1.
    precursor_mz2
        Precursor m/z of spec2.
    tolerance
        Maximum allowed m/z difference for a match. Shifted matches are only
        considered when the precursor difference exceeds it.
    mz_power
        Power to raise m/z values to.
    intensity_power
        Power to raise intensity values to.

    Returns
    -------
    score : float
        Modified cosine similarity score.
    matches : int
        Number of matched peak pairs with a nonzero product.

    Raises
    ------
    ValueError
        If either spectrum is not well-separated.
    """
    n1 = spec1.shape[0]
    n2 = spec2.shape[0]

    if n1 == 0 or n2 == 0:
        return 0.0, 0

    mz1 = spec1[:, 0]
    mz2 = spec2[:, 0]
    if not (_is_well_separated(mz1, tolerance) and _is_well_separated(mz2, tolerance)):
        raise ValueError(
            "Spectra must be well-separated (m/z gaps > 2 * tolerance), apply sirius_merge_close_peaks first."
        )

    products1, norm1 = _weighted_products(spec1, mz_power, intensity_power)
    products2, norm2 = _weighted_products(spec2, mz_power, intensity_power)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0, 0

    direct1 = np.empty(n1, dtype=np.int64)
    direct2 = np.empty(n2, dtype=np.int64)
    _linear_partners(mz1, mz2, tolerance, direct1, direct2)

    shifted1 = np.full(n1, -1, dtype=np.int64)
    shifted2 = np.full(n2, -1, dtype=np.int64)
    if abs(precursor_mz1 - precursor_mz2) > tolerance:
        _linear_partners(mz1 - precursor_mz1, mz2 - precursor_mz2, tolerance, shifted1, shifted2)
        # A pair within tolerance both directly and shifted is a single candidate.
        for i in range(n1):
            if shifted1[i] >= 0 and shifted1[i] == direct1[i]:
                shifted2[shifted1[i]] = -1
                shifted1[i] = -1

    max_edges = n1 + n2
    path_left = np.empty(max_edges, dtype=np.int64)
    path_right = np.empty(max_edges, dtype=np.int64)
    benefits = np.empty(max_edges, dtype=np.float64)
    dp = np.empty(max_edges + 1, dtype=np.float64)
    selected = np.empty(max_edges, dtype=np.bool_)
    visited1 = np.zeros(n1, dtype=np.bool_)
    visited2 = np.zeros(n2, dtype=np.bool_)

    matched_sum = 0.0
    matches = 0
    # Paths are acyclic, so walking from degree-one endpoints visits every edge.
    for start in range(max_edges):
        on_left = start < n1
        node = start if on_left else start - n1
        if on_left:
            if visited1[node] or (direct1[node] >= 0) == (shifted1[node] >= 0):
                continue
        elif visited2[node] or (direct2[node] >= 0) == (shifted2[node] >= 0):
            continue

        n_edges = _walk_path(
            node, on_left, direct1, shifted1, direct2, shifted2, visited1, visited2, path_left, path_right
        )
        for k in range(n_edges):
            benefits[k] = products1[path_left[k]] * products2[path_right[k]]
        _select_path_edges(n_edges, benefits, dp, selected)
        for k in range(n_edges):
            if selected[k] and benefits[k] != 0.0:
                matched_sum += benefits[k]
                matches += 1

    score = matched_sum / (norm1 * norm2)
    return score, matches
