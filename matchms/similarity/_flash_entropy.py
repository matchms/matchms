"""Dense indexed entropy scoring with deterministic one-to-one peak matching.

A query first visits fragment postings in increasing m/z order. In hybrid mode
it then visits neutral-loss postings in increasing loss order, ignoring peaks
already used by the fragment pass. Keeping these passes separate makes the
assignment rules visible without changing their order or floating-point math.
"""

import numpy as np
from numba import njit

from ._flash_tolerance import identity_allowed, window, within


@njit(cache=True, inline="always")
def xlog2(value: float) -> float:
    """Evaluate ``value * log2(value)`` with the continuous value zero at zero."""
    return value * np.log2(value) if value > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _accumulate_fragments(query, fragment, cursors, hybrid_state, tolerance, use_ppm, hybrid, epoch):
    """Assign direct fragment matches in ascending query and posting order.

    ``cursors`` groups score, last-used library peak and last-used query peak.
    ``hybrid_state`` holds the cross-pass consumed-peak and query-peak bitmaps.
    All buffers are mutated in place; none are cleared here.
    """
    query_mz, query_intensity = query
    fragment_mz, fragment_intensity, fragment_terms, fragment_ids, physical_ids = fragment
    scores, last_library_peak, last_query_peak = cursors
    used_peak_epochs, used_query_bits = hybrid_state

    for query_peak in range(query_mz.size):
        coordinate = np.float64(query_mz[query_peak])
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        query_term = xlog2(intensity)
        left, right = window(fragment_mz, coordinate, tolerance, use_ppm)
        for posting in range(left, right):
            library_id = fragment_ids[posting]
            physical_peak = physical_ids[posting]
            if last_query_peak[library_id] == query_peak or physical_peak <= last_library_peak[library_id]:
                continue
            reference_intensity = np.float64(fragment_intensity[posting])
            if reference_intensity <= 0.0:
                continue
            if not within(coordinate, np.float64(fragment_mz[posting]), tolerance, use_ppm):
                continue
            scores[library_id] += xlog2(intensity + reference_intensity) - query_term - fragment_terms[posting]
            last_query_peak[library_id] = query_peak
            last_library_peak[library_id] = physical_peak
            if hybrid:
                used_peak_epochs[physical_peak] = epoch
                used_query_bits[library_id, query_peak // 64] |= np.uint64(1) << np.uint64(query_peak % 64)


@njit(cache=True, nogil=True)
def _accumulate_losses(
    query,
    precursor_mz,
    neutral_loss,
    cursors,
    hybrid_state,
    n_library_peaks,
    tolerance,
    use_ppm,
    hybrid,
    epoch,
):
    """Assign losses in ascending loss order, respecting fragment-first hybrid use.

    Because query fragments are m/z-sorted, traversing them backwards visits
    the neutral losses in ascending order. The physical peak cursor therefore
    moves in the opposite direction to the fragment pass.
    """
    query_mz, query_intensity = query
    loss_mz, loss_intensity, loss_terms, loss_ids, physical_ids = neutral_loss
    scores, last_library_peak, last_query_peak = cursors
    used_peak_epochs, used_query_bits = hybrid_state

    last_library_peak[:] = n_library_peaks
    last_query_peak[:] = -1
    for query_peak in range(query_mz.size - 1, -1, -1):
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        coordinate = np.float64(precursor_mz) - np.float64(query_mz[query_peak])
        query_term = xlog2(intensity)
        left, right = window(loss_mz, coordinate, tolerance, use_ppm)
        for posting in range(left, right):
            library_id = loss_ids[posting]
            physical_peak = physical_ids[posting]
            if last_query_peak[library_id] == query_peak or physical_peak >= last_library_peak[library_id]:
                continue
            if hybrid:
                if used_peak_epochs[physical_peak] == epoch:
                    continue
                if used_query_bits[library_id, query_peak // 64] & (np.uint64(1) << np.uint64(query_peak % 64)):
                    continue
            reference_intensity = np.float64(loss_intensity[posting])
            if reference_intensity <= 0.0:
                continue
            if not within(coordinate, loss_mz[posting], tolerance, use_ppm):
                continue
            scores[library_id] += xlog2(intensity + reference_intensity) - query_term - loss_terms[posting]
            last_query_peak[library_id] = query_peak
            last_library_peak[library_id] = physical_peak


@njit(cache=True, nogil=True)
def _write_entropy_row(output_row, scores, query_precursor, library_precursors, tolerance, use_ppm):
    """Apply the optional precursor identity gate and cast into the output row."""
    for library_id in range(output_row.size):
        value = scores[library_id]
        if not identity_allowed(query_precursor, library_precursors[library_id], tolerance, use_ppm):
            value = 0.0
        output_row[library_id] = value


@njit(cache=True, nogil=True)
def entropy_rows(
    out: np.ndarray,
    q_offsets: np.ndarray,
    q_mz: np.ndarray,
    q_int: np.ndarray,
    q_pmz: np.ndarray,
    fragment: tuple,
    neutral_loss: tuple,
    lib_pmz: np.ndarray,
    n_lib_peaks: int,
    tolerance: float,
    use_ppm: bool,
    mode: int,
    identity_tol: float,
    identity_ppm: bool,
    start_row: int,
    stop_row: int,
) -> None:
    """Write exact entropy similarities for a block of queries to ``out``.

    Parameters
    ----------
    out
        Caller-owned ``(n_queries, n_references)`` score matrix. Only rows in
        ``[start_row, stop_row)`` are written.
    q_offsets, q_mz, q_int
        Packed, per-query m/z-sorted peaks: query ``r`` occupies the slice
        ``q_offsets[r]:q_offsets[r + 1]`` of both peak arrays. Intensities have
        already been entropy-weighted and normalized to sum to 0.5.
    q_pmz
        Query precursor masses, with missing values represented by ``NaN``.
    fragment, neutral_loss
        Each five-array posting bundle contains ``(m/z, intensity, xlog2 terms,
        library ID, physical peak ID)``, sorted by coordinate. The physical IDs
        identify original reference peaks in both posting views.
    lib_pmz
        Library precursor masses, with ``NaN`` for missing values.
    n_lib_peaks
        Total number of physical library peaks, used to initialize the
        descending neutral-loss cursor and hybrid peak-use tracking.
    tolerance, use_ppm
        Matching tolerance and whether it is symmetric ppm rather than Da.
    mode
        ``0`` for fragment, ``1`` for neutral loss, ``2`` for fragment-first
        hybrid matching. In hybrid mode each physical or query peak is used
        at most once for each spectrum pair across both passes.
    identity_tol, identity_ppm
        Optional precursor identity gate. Negative ``identity_tol`` disables
        the gate; otherwise both precursors must match within the tolerance.
    start_row, stop_row
        Start (inclusive) and stop (exclusive) query row IDs.

    Notes
    -----
    Accumulation occurs in float64, followed by assignment to the output dtype.
    Each call owns its scratch arrays; parallel callers must own disjoint rows.
    Traversal, one-to-one decisions and arithmetic order match the dense scorer
    prior to this readability refactor.
    """
    n_references = out.shape[1]
    scores = np.empty(n_references, dtype=np.float64)
    last_library_peak = np.empty(n_references, dtype=np.int64)
    last_query_peak = np.empty(n_references, dtype=np.int64)
    cursors = (scores, last_library_peak, last_query_peak)

    hybrid = mode == 2
    used_peak_epochs = np.zeros(n_lib_peaks if hybrid else 0, dtype=np.int64)
    max_query_peaks = 0
    for row in range(start_row, stop_row):
        max_query_peaks = max(max_query_peaks, q_offsets[row + 1] - q_offsets[row])
    used_query_bits = np.zeros(
        (n_references if hybrid else 0, (max_query_peaks + 63) // 64),
        dtype=np.uint64,
    )
    hybrid_state = (used_peak_epochs, used_query_bits)

    for row in range(start_row, stop_row):
        epoch = row - start_row + 1
        scores[:] = 0.0
        last_library_peak[:] = -1
        last_query_peak[:] = -1
        if hybrid:
            used_query_bits[:, :] = 0

        first_peak, stop_peak = q_offsets[row], q_offsets[row + 1]
        query = (q_mz[first_peak:stop_peak], q_int[first_peak:stop_peak])
        precursor_mz = q_pmz[row]

        if mode != 1:
            _accumulate_fragments(query, fragment, cursors, hybrid_state, tolerance, use_ppm, hybrid, epoch)
        if mode != 0 and np.isfinite(precursor_mz):
            _accumulate_losses(
                query,
                precursor_mz,
                neutral_loss,
                cursors,
                hybrid_state,
                n_lib_peaks,
                tolerance,
                use_ppm,
                hybrid,
                epoch,
            )
        _write_entropy_row(out[row], scores, precursor_mz, lib_pmz, identity_tol, identity_ppm)
