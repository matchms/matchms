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
def _accumulate_fragment_candidates(query, product_index, state, hybrid_state, tolerance, use_ppm, hybrid, epoch):
    """Accumulate direct edges and identify competing peaks in posting order."""
    query_mz, query_intensity = query
    product_mz, product_intensity, product_owner = product_index
    counts, conflicts, dot_products, last_query, last_peak = state
    peak_epochs, query_bits = hybrid_state

    for query_peak in range(query_mz.size):
        coordinate = np.float64(query_mz[query_peak])
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        left, right = window(product_mz, coordinate, tolerance, use_ppm)
        for posting in range(left, right):
            if not within(coordinate, np.float64(product_mz[posting]), tolerance, use_ppm):
                continue
            reference_id = product_owner[posting]
            counts[reference_id] += 1
            dot_products[reference_id] += intensity * np.float64(product_intensity[posting])
            if last_query[reference_id] == query_peak or posting <= last_peak[reference_id]:
                conflicts[reference_id] = True
            last_query[reference_id] = query_peak
            last_peak[reference_id] = posting
            if hybrid:
                peak_epochs[posting] = epoch
                query_bits[reference_id, query_peak // 64] |= np.uint64(1) << np.uint64(query_peak % 64)


@njit(cache=True, nogil=True)
def _accumulate_loss_candidates(
    query,
    precursor,
    product_index,
    loss_index,
    library_precursors,
    state,
    hybrid_state,
    tolerance,
    use_ppm,
    mode,
    epoch,
):
    """Accumulate neutral-loss edges; flag cross-pass conflicts in hybrid mode.

    The detection pass visits query peaks in *descending fragment order*.
    The later greedy fallback deliberately uses ascending fragment order.
    """
    query_mz, query_intensity = query
    _, product_intensity, _ = product_index
    loss_mz, loss_owner, loss_product_index = loss_index
    counts, conflicts, dot_products, last_query, last_peak = state
    peak_epochs, query_bits = hybrid_state

    last_query[:] = -1
    last_peak[:] = product_intensity.size
    for query_peak in range(query_mz.size - 1, -1, -1):
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        coordinate = precursor - np.float64(query_mz[query_peak])
        left, right = window(loss_mz, coordinate, tolerance, use_ppm)
        for loss_posting in range(left, right):
            if not within(coordinate, np.float64(loss_mz[loss_posting]), tolerance, use_ppm):
                continue
            reference_id = loss_owner[loss_posting]
            if mode == 2 and within(precursor, library_precursors[reference_id], tolerance, use_ppm):
                continue
            product_posting = loss_product_index[loss_posting]
            counts[reference_id] += 1
            dot_products[reference_id] += intensity * np.float64(product_intensity[product_posting])
            if mode == 2:
                word = query_peak // 64
                bit = np.uint64(1) << np.uint64(query_peak % 64)
                if peak_epochs[product_posting] == epoch or query_bits[reference_id, word] & bit:
                    conflicts[reference_id] = True
                peak_epochs[product_posting] = epoch
                query_bits[reference_id, word] |= bit
            elif last_query[reference_id] == query_peak or product_posting >= last_peak[reference_id]:
                conflicts[reference_id] = True
            last_query[reference_id] = query_peak
            last_peak[reference_id] = product_posting


@njit(cache=True, nogil=True)
def _normalize_unambiguous(scores, matched_peaks, row, state, query_norm, library_norms):
    """Finalize uncontested pairs and retain counts only for greedy conflicts.

    Conflicting columns keep their candidate count for the fallback. All other
    counts are reset to zero, exactly as in the dense implementation.
    """
    counts, conflicts, dot_products, _, _ = state
    store_matches = matched_peaks.shape[0] != 0
    need_fallback = False
    for reference_id in range(scores.shape[1]):
        if conflicts[reference_id]:
            need_fallback = True
            continue
        denominator = query_norm * library_norms[reference_id]
        if denominator > 0.0:
            scores[row, reference_id] = dot_products[reference_id] / denominator
        if store_matches:
            matched_peaks[row, reference_id] = counts[reference_id]
        counts[reference_id] = 0
    return need_fallback


@njit(cache=True, nogil=True)
def _apply_precursor_gate(scores, matched_peaks, row, query_precursor, reference_precursors, tolerance, use_ppm):
    """Remove pairs failing the optional precursor-identity filter."""
    store_matches = matched_peaks.shape[0] != 0
    for reference_id in range(scores.shape[1]):
        if not identity_allowed(query_precursor, reference_precursors[reference_id], tolerance, use_ppm):
            scores[row, reference_id] = 0.0
            if store_matches:
                matched_peaks[row, reference_id] = 0


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
    """Compute indexed cosine scores and optional accepted peak counts.

    Parameters
    ----------
    scores
        Caller-owned ``(n_queries, n_references)`` output, written only for
        query rows ``[start_row, stop_row)``.
    matched_peaks
        Caller-owned array of the same shape holding accepted one-to-one peak
        counts. Shape ``(0, 0)`` disables match-count storage.
    query_offsets, query_mz, query_int
        Spectrum-major query peak storage; query ``r`` is the slice
        ``query_offsets[r]:query_offsets[r + 1]`` in the m/z and intensity arrays.
    query_precursor, query_norms
        One precursor m/z (``NaN`` if unknown) and L2 intensity norm per query.
    peaks_mz, peaks_int, peaks_spec
        Globally m/z-sorted product postings, with each posting's source library
        spectrum ID in ``peaks_spec``.
    loss_mz, loss_spec, loss_product_index
        Sorted neutral-loss postings and their corresponding *product posting*
        indices. Required when ``mode`` is neutral loss or hybrid.
    library_precursor, library_norms
        Precursor m/z values and L2 norms for reference spectra.
    tolerance, use_ppm
        Inclusive fragment/loss matching tolerance, in Da or symmetric ppm.
    mode
        ``0`` fragment cosine; ``1`` neutral-loss cosine; ``2`` modified cosine
        with joint fragment and loss candidate assignment.
    identity_tolerance, identity_use_ppm
        Optional precursor-identity filter, with negative tolerance disabling it.
    start_row, stop_row
        Start (inclusive) and stop (exclusive) of query block.

    Notes
    -----
    The direct and loss sweeps first sum intensity products and detect competing
    peak matches. Uncontested pairs are normalized immediately. Only conflicting
    pairs use the product-sorted greedy assignment; matching order, fragment
    preference, float64 accumulation, and optional count semantics are preserved.
    Each call owns scratch arrays and must write disjoint rows when threaded.
    """
    n_references = scores.shape[1]
    counts = np.empty(n_references, dtype=np.int64)
    conflicts = np.empty(n_references, dtype=np.bool_)
    dot_products = np.empty(n_references, dtype=np.float64)
    last_query = np.empty(n_references, dtype=np.int64)
    last_peak = np.empty(n_references, dtype=np.int64)
    state = (counts, conflicts, dot_products, last_query, last_peak)
    product_index = (peaks_mz, peaks_int, peaks_spec)
    loss_index = (loss_mz, loss_spec, loss_product_index)

    max_query_peaks = 0
    for row in range(start_row, stop_row):
        max_query_peaks = max(max_query_peaks, query_offsets[row + 1] - query_offsets[row])
    hybrid = mode == 2
    peak_epochs = np.zeros(peaks_mz.size if hybrid else 0, dtype=np.int64)
    query_bits = np.zeros(
        (n_references if hybrid else 0, (max_query_peaks + 63) // 64),
        dtype=np.uint64,
    )
    hybrid_state = (peak_epochs, query_bits)

    for row in range(start_row, stop_row):
        first, stop = query_offsets[row], query_offsets[row + 1]
        scores[row, :] = 0.0
        if first == stop or np.isclose(query_norms[row], 0.0, rel_tol=0.0, abs_tol=1e-12):
            if matched_peaks.shape[0] != 0:
                matched_peaks[row, :] = 0
            continue

        epoch = row - start_row + 1
        counts[:] = 0
        conflicts[:] = False
        dot_products[:] = 0.0
        last_query[:] = -1
        last_peak[:] = -1
        if hybrid:
            query_bits[:, :] = 0
        precursor = np.float64(query_precursor[row])
        query = (query_mz[first:stop], query_int[first:stop])

        if mode != 1:
            _accumulate_fragment_candidates(
                query, product_index, state, hybrid_state, tolerance, use_ppm, hybrid, epoch
            )
        if mode != 0 and np.isfinite(precursor):
            _accumulate_loss_candidates(
                query,
                precursor,
                product_index,
                loss_index,
                library_precursor,
                state,
                hybrid_state,
                tolerance,
                use_ppm,
                mode,
                epoch,
            )

        need_fallback = _normalize_unambiguous(scores, matched_peaks, row, state, query_norms[row], library_norms)
        if need_fallback:
            offsets = np.empty(n_references + 1, dtype=np.int64)
            offsets[0] = 0
            for reference_id in range(n_references):
                offsets[reference_id + 1] = offsets[reference_id] + counts[reference_id]
            edge_query, edge_reference, products, is_fragment = _collect_conflicting_candidates(
                query_mz[first:stop],
                query_int[first:stop],
                precursor,
                peaks_mz,
                peaks_int,
                peaks_spec,
                loss_mz,
                loss_spec,
                loss_product_index,
                library_precursor,
                tolerance,
                use_ppm,
                mode,
                offsets,
                counts,
            )
            _resolve_conflicting_matches(
                scores,
                matched_peaks,
                row,
                stop - first,
                offsets,
                edge_query,
                edge_reference,
                products,
                is_fragment,
                library_norms,
                query_norms[row],
            )

        _apply_precursor_gate(
            scores, matched_peaks, row, precursor, library_precursor, identity_tolerance, identity_use_ppm
        )
