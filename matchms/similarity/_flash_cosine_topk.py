"""Exact candidate-aware top-k scoring for indexed cosine and modified cosine.

Reproduces ``cosine_rows``: first sum every candidate intensity product and
mark pairs with competing query/library peaks, then perform the same greedy
assignment *only* for conflicting reference spectra. Hybrid candidates from
fragment and neutral-loss windows compete jointly by intensity product.

Never constructs query-by-library score rows, clears entire library-sized
buffers per query, or scans all library IDs at row finalization. Conflicting
edges are grouped by a compact list of touched reference IDs rather than
using full-library prefix offsets. One workspace is retained per worker.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from ._flash_cosine import _FRAGMENT_SORT_BIAS
from ._flash_tolerance import identity_allowed, window, within
from ._flash_topk_common import _FlashTopKWorkspace, _register_touched
from ._topk import select_topk_candidates


class _CosineTopKWorkspace(_FlashTopKWorkspace):
    """Common touched-index buffers plus cosine matching and conflict cursors."""

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        super().__init__(n_references, n_peaks, dtype, hybrid)
        self.candidate_counts = np.empty(n_references, dtype=np.int64)
        self.conflicts = np.empty(n_references, dtype=np.bool_)
        self.last_query = np.empty(n_references, dtype=np.int64)
        self.last_peak = np.empty(n_references, dtype=np.int64)
        # A conflict column is read only after being initialized in this row.
        self.conflict_slot = np.empty(n_references, dtype=self.touched.dtype)


@njit(cache=True, inline="always")
def _touch_cosine(
    scores,
    counts,
    conflicts,
    epochs,
    last_query,
    last_peak,
    slots,
    touched,
    query_bits,
    n_touched,
    epoch,
    column,
    initial_peak,
):
    """Initialize a reference only once per query, regardless of hit frequency."""
    touched, query_bits, n_touched, new = _register_touched(
        epochs,
        slots,
        touched,
        query_bits,
        n_touched,
        epoch,
        column,
    )
    if new:
        scores[column] = 0.0
        counts[column] = 0
        conflicts[column] = False
        last_query[column] = -1
        last_peak[column] = initial_peak
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _collect_compact_conflicting_candidates(
    query_mz,
    query_int,
    query_precursor,
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
    epochs,
    epoch,
    conflicts,
    conflict_slot,
    offsets,
):
    """Generate only conflicting edges in the same order as the dense scorer.

    ``offsets`` partitions the arrays by conflict-list position, not reference
    ID. A conflict_slot is meaningful only for a column touched in this epoch
    whose ``conflicts`` value is true.
    """
    n_edges = offsets[-1]
    query_indices = np.empty(n_edges, dtype=np.int64)
    library_indices = np.empty(n_edges, dtype=np.int64)
    products = np.empty(n_edges, dtype=np.float64)
    is_fragment = np.empty(n_edges, dtype=np.uint8)
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
                if epochs[column] != epoch or not conflicts[column]:
                    continue
                if not within(coordinate, np.float64(peaks_mz[peak_index]), tolerance, use_ppm):
                    continue
                position = positions[conflict_slot[column]]
                query_indices[position] = query_index
                library_indices[position] = peak_index
                products[position] = intensity * np.float64(peaks_int[peak_index])
                is_fragment[position] = 1
                positions[conflict_slot[column]] += 1

    if mode != 0 and np.isfinite(query_precursor):
        # The original fallback enumerates losses by ascending *fragment* m/z,
        # unlike the initial conflict-detection sweep (descending fragments).
        for query_index in range(query_mz.size):
            intensity = np.float64(query_int[query_index])
            if intensity <= 0.0:
                continue
            coordinate = query_precursor - np.float64(query_mz[query_index])
            left, right = window(loss_mz, coordinate, tolerance, use_ppm)
            for loss_index in range(left, right):
                column = loss_spec[loss_index]
                if epochs[column] != epoch or not conflicts[column]:
                    continue
                if not within(coordinate, np.float64(loss_mz[loss_index]), tolerance, use_ppm):
                    continue
                if mode == 2 and within(query_precursor, library_precursor[column], tolerance, use_ppm):
                    continue
                peak_index = loss_product_index[loss_index]
                position = positions[conflict_slot[column]]
                query_indices[position] = query_index
                library_indices[position] = peak_index
                products[position] = intensity * np.float64(peaks_int[peak_index])
                is_fragment[position] = 0
                positions[conflict_slot[column]] += 1
    return query_indices, library_indices, products, is_fragment


@njit(cache=True, nogil=True)
def _resolve_compact_conflicts(
    scores,
    accepted_counts,
    conflict_ids,
    offsets,
    query_indices,
    library_indices,
    products,
    is_fragment,
    library_norms,
    query_norm,
    n_query_peaks,
):
    """Use the original greedy product-sorted assignment, one pair at a time."""
    used_query = np.empty(n_query_peaks, dtype=np.uint8)
    for slot in range(conflict_ids.size):
        column = conflict_ids[slot]
        start, stop = offsets[slot], offsets[slot + 1]
        size = stop - start
        if size == 0:
            continue
        keys = np.empty(size, dtype=np.float64)
        for position in range(size):
            edge = start + position
            keys[position] = products[edge] + (_FRAGMENT_SORT_BIAS if is_fragment[edge] else 0.0)
        # Identical operation, keys and within-pair edge order as the dense path.
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
        scores[column] = dot_product / denominator if denominator > 0.0 else 0.0
        accepted_counts[column] = n_used_library


@njit(cache=True, nogil=True)
def _cosine_topk_row(
    q_mz,
    q_int,
    q_precursor,
    query_norm,
    peaks_mz,
    peaks_int,
    peaks_spec,
    loss_mz,
    loss_spec,
    loss_product_index,
    library_precursor,
    library_norms,
    tolerance,
    use_ppm,
    mode,
    identity_tol,
    identity_ppm,
    scores,
    counts,
    conflicts,
    epochs,
    last_query,
    last_peak,
    slots,
    touched,
    query_bits,
    peak_word_epochs,
    peak_word_bits,
    conflict_slot,
    epoch,
):
    """Score all touched candidate pairs; return growable buffers and count."""
    n_touched = 0
    n_lib_peaks = peaks_mz.size
    if not q_mz.size or query_norm == 0.0:
        return touched, query_bits, n_touched

    if mode != 1:
        for query_index in range(q_mz.size):
            coordinate = np.float64(q_mz[query_index])
            intensity = np.float64(q_int[query_index])
            if intensity <= 0.0:
                continue
            left, right = window(peaks_mz, coordinate, tolerance, use_ppm)
            for peak_index in range(left, right):
                if not within(coordinate, np.float64(peaks_mz[peak_index]), tolerance, use_ppm):
                    continue
                column = peaks_spec[peak_index]
                touched, query_bits, n_touched = _touch_cosine(
                    scores,
                    counts,
                    conflicts,
                    epochs,
                    last_query,
                    last_peak,
                    slots,
                    touched,
                    query_bits,
                    n_touched,
                    epoch,
                    column,
                    -1,
                )
                counts[column] += 1
                scores[column] += intensity * np.float64(peaks_int[peak_index])
                if last_query[column] == query_index or peak_index <= last_peak[column]:
                    conflicts[column] = True
                last_query[column] = query_index
                last_peak[column] = peak_index
                if mode == 2:
                    word = peak_index // 64
                    if peak_word_epochs[word] != epoch:
                        peak_word_epochs[word] = epoch
                        peak_word_bits[word] = np.uint64(0)
                    peak_word_bits[word] |= np.uint64(1) << np.uint64(peak_index % 64)
                    query_bits[slots[column], query_index // 64] |= np.uint64(1) << np.uint64(query_index % 64)

    if mode != 0 and np.isfinite(q_precursor):
        for position in range(n_touched):
            column = touched[position]
            last_query[column] = -1
            last_peak[column] = n_lib_peaks
        # Preserve the *descending fragment* order of the dense detection pass.
        for query_index in range(q_mz.size - 1, -1, -1):
            intensity = np.float64(q_int[query_index])
            if intensity <= 0.0:
                continue
            loss = np.float64(q_precursor) - np.float64(q_mz[query_index])
            left, right = window(loss_mz, loss, tolerance, use_ppm)
            for loss_index in range(left, right):
                if not within(loss, np.float64(loss_mz[loss_index]), tolerance, use_ppm):
                    continue
                column = loss_spec[loss_index]
                if mode == 2 and within(q_precursor, library_precursor[column], tolerance, use_ppm):
                    continue
                peak_index = loss_product_index[loss_index]
                touched, query_bits, n_touched = _touch_cosine(
                    scores,
                    counts,
                    conflicts,
                    epochs,
                    last_query,
                    last_peak,
                    slots,
                    touched,
                    query_bits,
                    n_touched,
                    epoch,
                    column,
                    n_lib_peaks,
                )
                counts[column] += 1
                scores[column] += intensity * np.float64(peaks_int[peak_index])
                if mode == 2:
                    word = peak_index // 64
                    peak_used = (
                        peak_word_epochs[word] == epoch
                        and (peak_word_bits[word] & (np.uint64(1) << np.uint64(peak_index % 64))) != 0
                    )
                    query_used = (
                        query_bits[slots[column], query_index // 64] & (np.uint64(1) << np.uint64(query_index % 64))
                    ) != 0
                    if peak_used or query_used:
                        conflicts[column] = True
                    if peak_word_epochs[word] != epoch:
                        peak_word_epochs[word] = epoch
                        peak_word_bits[word] = np.uint64(0)
                    peak_word_bits[word] |= np.uint64(1) << np.uint64(peak_index % 64)
                    query_bits[slots[column], query_index // 64] |= np.uint64(1) << np.uint64(query_index % 64)
                elif last_query[column] == query_index or peak_index >= last_peak[column]:
                    conflicts[column] = True
                last_query[column] = query_index
                last_peak[column] = peak_index

    # Normalize nonconflicting pairs; build a *compact* list of conflicts.
    conflict_ids = np.empty(n_touched, dtype=touched.dtype)
    n_conflicts = 0
    for position in range(n_touched):
        column = touched[position]
        if conflicts[column]:
            conflict_slot[column] = n_conflicts
            conflict_ids[n_conflicts] = column
            n_conflicts += 1
        else:
            denominator = query_norm * library_norms[column]
            scores[column] = scores[column] / denominator if denominator > 0.0 else 0.0

    if n_conflicts:
        conflict_ids = conflict_ids[:n_conflicts]
        offsets = np.empty(n_conflicts + 1, dtype=np.int64)
        offsets[0] = 0
        for slot in range(n_conflicts):
            offsets[slot + 1] = offsets[slot] + counts[conflict_ids[slot]]
        query_indices, library_indices, products, is_fragment = _collect_compact_conflicting_candidates(
            q_mz,
            q_int,
            q_precursor,
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
            epochs,
            epoch,
            conflicts,
            conflict_slot,
            offsets,
        )
        _resolve_compact_conflicts(
            scores,
            counts,
            conflict_ids,
            offsets,
            query_indices,
            library_indices,
            products,
            is_fragment,
            library_norms,
            query_norm,
            q_mz.size,
        )

    if identity_tol >= 0.0:
        for position in range(n_touched):
            column = touched[position]
            if not identity_allowed(q_precursor, library_precursor[column], identity_tol, identity_ppm):
                scores[column] = 0.0
                counts[column] = 0
    return touched, query_bits, n_touched


def _score_cosine_topk_block(
    prepared,
    library,
    workspace: _CosineTopKWorkspace,
    indices: np.ndarray,
    scores: np.ndarray,
    valid_counts: np.ndarray,
    matches: np.ndarray | None,
    first_row: int,
    last_row: int,
    k: int,
    min_score: float,
    mode: int,
    tolerance: float,
    use_ppm: bool,
    identity_tol: float,
    identity_ppm: bool,
) -> None:
    """Accumulate and select exact best candidates in disjoint output rows."""
    for row in range(first_row, last_row):
        first = int(prepared.spec_offsets[row])
        last = int(prepared.spec_offsets[row + 1])
        if mode == 2:
            workspace.ensure_query_words(last - first)
        epoch = workspace.next_epoch()
        touched, query_bits, n_touched = _cosine_topk_row(
            prepared.spec_mz[first:last],
            prepared.spec_int[first:last],
            np.float64(prepared.precursor_mz[row]),
            np.float64(prepared.spec_l2[row]),
            library.peaks_mz,
            library.peaks_int,
            library.peaks_spec_idx,
            library.nl_mz,
            library.nl_spec_idx,
            library.nl_product_idx,
            library.precursor_mz,
            library.spec_l2,
            tolerance,
            use_ppm,
            mode,
            identity_tol,
            identity_ppm,
            workspace.scores,
            workspace.candidate_counts,
            workspace.conflicts,
            workspace.epochs,
            workspace.last_query,
            workspace.last_peak,
            workspace.slots,
            workspace.touched,
            workspace.query_bits,
            workspace.peak_word_epochs,
            workspace.peak_word_bits,
            workspace.conflict_slot,
            epoch,
        )
        workspace.touched = touched
        workspace.query_bits = query_bits
        candidates = touched[:n_touched]
        if workspace.selection_scores is not workspace.scores:
            workspace.selection_scores[candidates] = workspace.scores[candidates]
        ids, values = select_topk_candidates(
            candidates,
            workspace.selection_scores,
            k,
            min_score,
        )
        n_hits = ids.size
        indices[row, :n_hits] = ids
        scores[row, :n_hits] = values
        valid_counts[row] = n_hits
        if matches is not None:
            matches[row, :n_hits] = workspace.candidate_counts[ids]
