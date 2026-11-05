"""Indexed cosine scoring with one-to-one, intensity-product greedy assignment.

A posting is one library peak in a globally m/z-sorted index. Independent
candidate matches can be accumulated directly. If any candidate reuses a query
or library peak, all candidates for that library spectrum are passed to the
ordered greedy assignment. Only conflicting spectrum pairs require sorting.

The candidate counts already needed by the conflict-resolution step are also
the matched-peak counts for unambiguous spectrum pairs. A separate matching pass
is therefore unnecessary when counts are requested.
"""
import numpy as np
from numba import njit
from ._flash_tolerance import identity_allowed, window, within


# Preserve the established Flash cosine ordering, including its small additive
# preference for direct fragments. This is not a strict lexicographic tie-break.
_FRAGMENT_SORT_BIAS = 1e-12


@njit(cache=True, nogil=True)
def _collect_conflicting_candidates(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    query_precursor: float,
    peaks_mz: np.ndarray,
    peaks_int: np.ndarray,
    peaks_spec: np.ndarray,
    loss_mz: np.ndarray,
    loss_spec: np.ndarray,
    loss_product_index: np.ndarray,
    library_precursor: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    mode: int,
    offsets: np.ndarray,
    counts: np.ndarray,
) -> tuple:
    """Collect candidate edges for conflicting library spectra only.

    ``counts`` is zero for nonconflicting spectra. ``offsets`` contains its
    prefix sums and partitions the returned arrays by library spectrum. Each
    edge records a query-peak index, a global product-peak index, an intensity
    product, and whether it is a direct fragment match.

    Candidate enumeration must remain identical to the assignment convention:
    query peaks are visited in ascending fragment m/z order, first for direct
    matches and then for loss matches. In hybrid mode, loss candidates are
    omitted when the precursor difference is within the matching tolerance.
    """
    n_candidates = offsets[-1]
    query_indices = np.empty(n_candidates, dtype=np.int64)
    library_indices = np.empty(n_candidates, dtype=np.int64)
    products = np.empty(n_candidates, dtype=np.float64)
    is_fragment = np.empty(n_candidates, dtype=np.uint8)
    positions = offsets[:-1].copy()

    if mode != 1:
        for query_index in range(query_mz.size):
            intensity = np.float64(query_int[query_index])
            if intensity <= 0.0:
                continue
            coordinate = np.float64(query_mz[query_index])
            left, right = window(peaks_mz, coordinate, tolerance, use_ppm)
            for peak_index in range(left, right):
                column = peaks_spec[peak_index]
                if counts[column] == 0:
                    continue
                if not within(coordinate, np.float64(peaks_mz[peak_index]), tolerance, use_ppm):
                    continue
                position = positions[column]
                query_indices[position] = query_index
                library_indices[position] = peak_index
                products[position] = intensity * np.float64(peaks_int[peak_index])
                is_fragment[position] = 1
                positions[column] += 1

    if mode != 0 and np.isfinite(query_precursor):
        for query_index in range(query_mz.size):
            intensity = np.float64(query_int[query_index])
            if intensity <= 0.0:
                continue
            coordinate = query_precursor - np.float64(query_mz[query_index])
            left, right = window(loss_mz, coordinate, tolerance, use_ppm)
            for loss_index in range(left, right):
                column = loss_spec[loss_index]
                if counts[column] == 0:
                    continue
                if not within(coordinate, np.float64(loss_mz[loss_index]), tolerance, use_ppm):
                    continue
                if mode == 2 and within(query_precursor, library_precursor[column], tolerance, use_ppm):
                    continue
                peak_index = loss_product_index[loss_index]
                position = positions[column]
                query_indices[position] = query_index
                library_indices[position] = peak_index
                products[position] = intensity * np.float64(peaks_int[peak_index])
                is_fragment[position] = 0
                positions[column] += 1

    return query_indices, library_indices, products, is_fragment


@njit(cache=True, nogil=True)
def _resolve_conflicting_matches(
    scores: np.ndarray,
    matched_peaks: np.ndarray,
    row: int,
    n_query_peaks: int,
    offsets: np.ndarray,
    query_indices: np.ndarray,
    library_indices: np.ndarray,
    products: np.ndarray,
    is_fragment: np.ndarray,
    library_norms: np.ndarray,
    query_norm: float,
) -> None:
    """Overwrite conflicting pairs with greedy cosine scores and match counts.

    For each library spectrum, candidates are sorted by descending intensity
    product plus the direct-fragment bias. A candidate is accepted only if both
    peaks are unused. The denominator uses all prepared peaks, not only the
    matched ones. Counts refer to accepted one-to-one assignments, never the
    number of candidate edges.

    The number of occupied entries in ``used_library`` is already the match
    count, so counting needs no additional per-edge operation. A zero-row
    ``matched_peaks`` array disables output of those counts. Scores for columns
    with no candidates in ``offsets`` are untouched.
    """
    store_matches = matched_peaks.shape[0] != 0
    used_query = np.empty(n_query_peaks, dtype=np.uint8)
    for column in range(scores.shape[1]):
        start, stop = offsets[column], offsets[column + 1]
        size = stop - start
        if size == 0:
            continue

        keys = np.empty(size, dtype=np.float64)
        for position in range(size):
            edge = start + position
            keys[position] = products[edge] + (_FRAGMENT_SORT_BIAS if is_fragment[edge] else 0.0)
        order = np.argsort(-keys)

        used_query[:] = 0
        used_library = np.empty(size, dtype=np.int64)
        n_used_library = 0
        dot_product = 0.0
        for position in order:
            edge = start + position
            query_index = query_indices[edge]
            library_index = library_indices[edge]
            if used_query[query_index]:
                continue
            seen = False
            for used_index in range(n_used_library):
                if used_library[used_index] == library_index:
                    seen = True
                    break
            if seen:
                continue
            used_query[query_index] = 1
            used_library[n_used_library] = library_index
            n_used_library += 1
            dot_product += products[edge]

        denominator = query_norm * library_norms[column]
        scores[row, column] = dot_product / denominator if denominator > 0.0 else 0.0
        if store_matches:
            matched_peaks[row, column] = n_used_library


@njit(cache=True, nogil=True)
def cosine_rows(
    scores: np.ndarray,
    matched_peaks: np.ndarray,
    query_offsets: np.ndarray,
    query_mz: np.ndarray,
    query_int: np.ndarray,
    query_precursor: np.ndarray,
    query_norms: np.ndarray,
    peaks_mz: np.ndarray,
    peaks_int: np.ndarray,
    peaks_spec: np.ndarray,
    loss_mz: np.ndarray,
    loss_spec: np.ndarray,
    loss_product_index: np.ndarray,
    library_precursor: np.ndarray,
    library_norms: np.ndarray,
    tolerance: float,
    use_ppm: bool,
    mode: int,
    identity_tolerance: float,
    identity_use_ppm: bool,
    start_row: int,
    stop_row: int,
) -> None:
    """Write cosine scores and optional counts for a contiguous query block.

    Modes are 0 (fragment), 1 (neutral loss), and 2 (hybrid/modified cosine).
    Hybrid candidates compete by intensity product; fragment matches do not
    consume peaks before the neutral-loss candidates are considered.

    The first pass accumulates intensity products, counts candidates, and detects
    reused peaks. Independent candidates are already valid one-to-one matches.
    Conflicting columns alone are materialized and greedily assigned afterwards.

    Scratch arrays are private to this call. Concurrent calls must write disjoint
    output rows. ``matched_peaks.shape == (0, 0)`` disables count output without
    altering the candidate enumeration or the resulting score.
    """
    n_library = scores.shape[1]
    n_library_peaks = peaks_mz.size
    store_matches = matched_peaks.shape[0] != 0
    candidate_counts = np.empty(n_library, dtype=np.int64)
    conflicts = np.empty(n_library, dtype=np.bool_)
    dot_products = np.empty(n_library, dtype=np.float64)
    last_query = np.empty(n_library, dtype=np.int64)
    last_peak = np.empty(n_library, dtype=np.int64)
    max_query_peaks = 0
    for row in range(start_row, stop_row):
        max_query_peaks = max(max_query_peaks, query_offsets[row + 1] - query_offsets[row])

    # Within one coordinate pass, monotone cursors detect reused peaks. Hybrid
    # also needs cross-pass state: one epoch per library peak and one query-bit
    # mask per library spectrum. Epochs avoid clearing the peak-sized array for
    # every query. The word dimension supports spectra with more than 64 peaks.
    peak_epochs = np.zeros(n_library_peaks if mode == 2 else 0, dtype=np.int64)
    query_bits = np.zeros(
        (n_library if mode == 2 else 0, (max_query_peaks + 63) // 64), dtype=np.uint64,
    )

    for row in range(start_row, stop_row):
        start, stop = query_offsets[row], query_offsets[row + 1]
        scores[row, :] = 0.0
        if start == stop or query_norms[row] == 0.0:
            if store_matches:
                matched_peaks[row, :] = 0
            continue

        epoch = row - start_row + 1
        candidate_counts[:] = 0
        conflicts[:] = False
        dot_products[:] = 0.0
        last_query[:] = -1
        last_peak[:] = -1
        if mode == 2:
            query_bits[:, :] = 0
        precursor = np.float64(query_precursor[row])

        if mode != 1:
            for query_index in range(stop - start):
                coordinate = np.float64(query_mz[start + query_index])
                intensity = np.float64(query_int[start + query_index])
                if intensity <= 0.0:
                    continue
                left, right = window(peaks_mz, coordinate, tolerance, use_ppm)
                for peak_index in range(left, right):
                    if not within(coordinate, np.float64(peaks_mz[peak_index]), tolerance, use_ppm):
                        continue
                    column = peaks_spec[peak_index]
                    candidate_counts[column] += 1
                    dot_products[column] += intensity * np.float64(peaks_int[peak_index])
                    if last_query[column] == query_index or peak_index <= last_peak[column]:
                        conflicts[column] = True
                    last_query[column] = query_index
                    last_peak[column] = peak_index
                    if mode == 2:
                        peak_epochs[peak_index] = epoch
                        query_bits[column, query_index // 64] |= np.uint64(1) << np.uint64(query_index % 64)

        if mode != 0 and np.isfinite(precursor):
            last_query[:] = -1
            last_peak[:] = n_library_peaks
            # Descending fragments give ascending neutral losses. Candidate
            # enumeration for the greedy fallback is separate and remains in
            # ascending fragment order, which preserves equal-weight choices.
            for query_index in range(stop - start - 1, -1, -1):
                intensity = np.float64(query_int[start + query_index])
                if intensity <= 0.0:
                    continue
                loss = precursor - np.float64(query_mz[start + query_index])
                left, right = window(loss_mz, loss, tolerance, use_ppm)
                for loss_index in range(left, right):
                    if not within(loss, np.float64(loss_mz[loss_index]), tolerance, use_ppm):
                        continue
                    column = loss_spec[loss_index]
                    if mode == 2 and within(precursor, library_precursor[column], tolerance, use_ppm):
                        continue
                    peak_index = loss_product_index[loss_index]
                    candidate_counts[column] += 1
                    dot_products[column] += intensity * np.float64(peaks_int[peak_index])
                    if mode == 2:
                        word = query_index // 64
                        bit = np.uint64(1) << np.uint64(query_index % 64)
                        if peak_epochs[peak_index] == epoch or query_bits[column, word] & bit:
                            conflicts[column] = True
                        peak_epochs[peak_index] = epoch
                        query_bits[column, word] |= bit
                    elif last_query[column] == query_index or peak_index >= last_peak[column]:
                        conflicts[column] = True
                    last_query[column] = query_index
                    last_peak[column] = peak_index

        need_fallback = False
        for column in range(n_library):
            if conflicts[column]:
                need_fallback = True
                continue
            denominator = query_norms[row] * library_norms[column]
            if denominator > 0.0:
                scores[row, column] = dot_products[column] / denominator
            if store_matches:
                matched_peaks[row, column] = candidate_counts[column]
            # Only conflicting columns participate in the second pass.
            candidate_counts[column] = 0

        if need_fallback:
            offsets = np.empty(n_library + 1, dtype=np.int64)
            offsets[0] = 0
            for column in range(n_library):
                offsets[column + 1] = offsets[column] + candidate_counts[column]
            query_indices, library_indices, products, is_fragment = _collect_conflicting_candidates(
                query_mz[start:stop], query_int[start:stop], precursor,
                peaks_mz, peaks_int, peaks_spec, loss_mz, loss_spec, loss_product_index,
                library_precursor, tolerance, use_ppm, mode, offsets, candidate_counts,
            )
            _resolve_conflicting_matches(
                scores, matched_peaks, row, stop - start, offsets, query_indices,
                library_indices, products, is_fragment, library_norms, query_norms[row],
            )

        for column in range(n_library):
            if not identity_allowed(precursor, library_precursor[column], identity_tolerance, identity_use_ppm):
                scores[row, column] = 0.0
                if store_matches:
                    matched_peaks[row, column] = 0
