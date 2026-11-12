"""Exact candidate-aware cosine and modified-cosine top-k scoring.

Unlike dense cosine scoring, only reference spectra encountered through peak
postings are initialized and ranked. Candidate collection has two passes:
fragment edges, then neutral-loss edges. The passes detect conflicts, but do
not assign matches. Only spectra with conflicting edges run the original greedy
intensity-product assignment, with the same small fragment preference.

For Numba efficiency, the private helpers accept small *bundles* of related
arrays instead of repeating the entire 20-plus-argument row-kernel signature.
The bundles are unpacked with descriptive names at each helper's entry.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit
from ._flash_cosine import _FRAGMENT_SORT_BIAS
from ._flash_tolerance import identity_allowed, window, within
from ._flash_topk_common import _FlashTopKWorkspace, _register_touched
from ._topk import select_topk_candidates


if TYPE_CHECKING:
    from ._flash_prepared import PreparedSpectra
    from .flash_index import _CosineKernelData


class _CosineTopKWorkspace(_FlashTopKWorkspace):
    """Per-worker cosine scratch buffers, including greedy-conflict bookkeeping.

    Inherited epoch stamps mark initialized reference IDs. Candidate counts,
    conflict flags, and last-used peak/query cursors are meaningful only for
    references touched during the current epoch. ``conflict_slot`` maps a
    conflicting reference ID to its compact candidate-edge partition.
    """

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        super().__init__(n_references, n_peaks, dtype, hybrid)
        self.candidate_counts = np.empty(n_references, dtype=np.int64)
        self.conflicts = np.empty(n_references, dtype=np.bool_)
        self.last_query = np.empty(n_references, dtype=np.int64)
        self.last_peak = np.empty(n_references, dtype=np.int64)
        self.conflict_slot = np.empty(n_references, dtype=self.touched.dtype)


@njit(cache=True, inline="always")
def _touch_cosine(state, touched, query_bits, n_touched, epoch, reference_id, initial_peak):
    """Register a candidate once, initializing its dot product and cursors."""
    scores, counts, conflicts, epochs, last_query, last_peak, slots, _, _, _ = state
    touched, query_bits, n_touched, is_new = _register_touched(
        epochs,
        slots,
        touched,
        query_bits,
        n_touched,
        epoch,
        reference_id,
    )
    if is_new:
        scores[reference_id] = 0.0
        counts[reference_id] = 0
        conflicts[reference_id] = False
        last_query[reference_id] = -1
        last_peak[reference_id] = initial_peak
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _find_fragment_candidates(
    query, product_index, state, touched, query_bits, n_touched, tolerance, use_ppm, hybrid, epoch
):
    """Accumulate all direct edges and flag repeated query or physical peaks."""
    query_mz, query_intensity, _, _ = query
    product_mz, product_intensity, product_owner = product_index
    scores, counts, conflicts, _, last_query, last_peak, slots, peak_word_epochs, peak_word_bits, _ = state

    for query_peak in range(query_mz.size):
        coordinate = np.float64(query_mz[query_peak])
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        first, stop = window(product_mz, coordinate, tolerance, use_ppm)
        for posting in range(first, stop):
            if not within(coordinate, np.float64(product_mz[posting]), tolerance, use_ppm):
                continue
            reference_id = product_owner[posting]
            touched, query_bits, n_touched = _touch_cosine(
                state,
                touched,
                query_bits,
                n_touched,
                epoch,
                reference_id,
                -1,
            )
            counts[reference_id] += 1
            scores[reference_id] += intensity * np.float64(product_intensity[posting])
            if last_query[reference_id] == query_peak or posting <= last_peak[reference_id]:
                conflicts[reference_id] = True
            last_query[reference_id] = query_peak
            last_peak[reference_id] = posting
            if hybrid:
                word = posting // 64
                if peak_word_epochs[word] != epoch:
                    peak_word_epochs[word] = epoch
                    peak_word_bits[word] = np.uint64(0)
                peak_word_bits[word] |= np.uint64(1) << np.uint64(posting % 64)
                query_bits[slots[reference_id], query_peak // 64] |= np.uint64(1) << np.uint64(query_peak % 64)
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _find_loss_candidates(
    query,
    product_index,
    loss_index,
    reference_precursors,
    state,
    touched,
    query_bits,
    n_touched,
    tolerance,
    use_ppm,
    mode,
    epoch,
):
    """Detect neutral-loss edges and cross-pass competition in modified cosine.

    This initial detection visits queries from highest to lowest fragment m/z.
    The greedy fallback later enumerates loss edges in ascending fragment order,
    matching the historical dense algorithm's tie ordering.
    """
    query_mz, query_intensity, precursor_mz, _ = query
    _, product_intensity, _ = product_index
    loss_mz, loss_owner, loss_product_index = loss_index
    scores, counts, conflicts, _, last_query, last_peak, slots, peak_word_epochs, peak_word_bits, _ = state

    for position in range(n_touched):
        reference_id = touched[position]
        last_query[reference_id] = -1
        last_peak[reference_id] = product_intensity.size

    for query_peak in range(query_mz.size - 1, -1, -1):
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        coordinate = np.float64(precursor_mz) - np.float64(query_mz[query_peak])
        first, stop = window(loss_mz, coordinate, tolerance, use_ppm)
        for loss_posting in range(first, stop):
            if not within(coordinate, np.float64(loss_mz[loss_posting]), tolerance, use_ppm):
                continue
            reference_id = loss_owner[loss_posting]
            if mode == 2 and within(precursor_mz, reference_precursors[reference_id], tolerance, use_ppm):
                continue
            product_posting = loss_product_index[loss_posting]
            touched, query_bits, n_touched = _touch_cosine(
                state,
                touched,
                query_bits,
                n_touched,
                epoch,
                reference_id,
                product_intensity.size,
            )
            counts[reference_id] += 1
            scores[reference_id] += intensity * np.float64(product_intensity[product_posting])
            if mode == 2:
                word = product_posting // 64
                peak_used = (
                    peak_word_epochs[word] == epoch
                    and (peak_word_bits[word] & (np.uint64(1) << np.uint64(product_posting % 64))) != 0
                )
                query_used = (
                    query_bits[slots[reference_id], query_peak // 64] & (np.uint64(1) << np.uint64(query_peak % 64))
                ) != 0
                if peak_used or query_used:
                    conflicts[reference_id] = True
                if peak_word_epochs[word] != epoch:
                    peak_word_epochs[word] = epoch
                    peak_word_bits[word] = np.uint64(0)
                peak_word_bits[word] |= np.uint64(1) << np.uint64(product_posting % 64)
                query_bits[slots[reference_id], query_peak // 64] |= np.uint64(1) << np.uint64(query_peak % 64)
            elif last_query[reference_id] == query_peak or product_posting >= last_peak[reference_id]:
                conflicts[reference_id] = True
            last_query[reference_id] = query_peak
            last_peak[reference_id] = product_posting
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _collect_conflict_edges(query, product_index, loss_index, reference_precursors, matching, state, epoch, offsets):
    """Collect only conflicting edges, preserving dense greedy enumeration order.

    ``offsets`` partitions edges by *conflict-list position*, rather than by
    reference ID. ``conflict_slot`` is valid only for IDs marked conflicting in
    this epoch. The returned arrays align query IDs, physical product posting
    IDs, products and fragment flags.
    """
    query_mz, query_intensity, precursor_mz, _ = query
    product_mz, product_intensity, product_owner = product_index
    loss_mz, loss_owner, loss_product_index = loss_index
    mode, tolerance, use_ppm, _, _ = matching
    _, _, conflicts, epochs, _, _, _, _, _, conflict_slot = state

    n_edges = offsets[-1]
    query_indices = np.empty(n_edges, dtype=np.int64)
    library_indices = np.empty(n_edges, dtype=np.int64)
    products = np.empty(n_edges, dtype=np.float64)
    is_fragment = np.empty(n_edges, dtype=np.uint8)
    positions = offsets[:-1].copy()

    if mode != 1:
        for query_peak in range(query_mz.size):
            intensity = np.float64(query_intensity[query_peak])
            if intensity <= 0.0:
                continue
            coordinate = np.float64(query_mz[query_peak])
            left, right = window(product_mz, coordinate, tolerance, use_ppm)
            for posting in range(left, right):
                reference_id = product_owner[posting]
                if epochs[reference_id] != epoch or not conflicts[reference_id]:
                    continue
                if not within(coordinate, np.float64(product_mz[posting]), tolerance, use_ppm):
                    continue
                slot = conflict_slot[reference_id]
                position = positions[slot]
                query_indices[position] = query_peak
                library_indices[position] = posting
                products[position] = intensity * np.float64(product_intensity[posting])
                is_fragment[position] = 1
                positions[slot] += 1

    if mode != 0 and np.isfinite(precursor_mz):
        for query_peak in range(query_mz.size):
            intensity = np.float64(query_intensity[query_peak])
            if intensity <= 0.0:
                continue
            coordinate = precursor_mz - np.float64(query_mz[query_peak])
            left, right = window(loss_mz, coordinate, tolerance, use_ppm)
            for loss_posting in range(left, right):
                reference_id = loss_owner[loss_posting]
                if epochs[reference_id] != epoch or not conflicts[reference_id]:
                    continue
                if not within(coordinate, np.float64(loss_mz[loss_posting]), tolerance, use_ppm):
                    continue
                if mode == 2 and within(precursor_mz, reference_precursors[reference_id], tolerance, use_ppm):
                    continue
                product_posting = loss_product_index[loss_posting]
                slot = conflict_slot[reference_id]
                position = positions[slot]
                query_indices[position] = query_peak
                library_indices[position] = product_posting
                products[position] = intensity * np.float64(product_intensity[product_posting])
                is_fragment[position] = 0
                positions[slot] += 1
    return query_indices, library_indices, products, is_fragment


@njit(cache=True, nogil=True)
def _resolve_compact_conflicts(scores, counts, conflict_ids, offsets, edges, library_norms, query_norm, n_query_peaks):
    """Apply unchanged product-sorted greedy one-to-one assignment to conflicts."""
    query_indices, library_indices, products, is_fragment = edges
    used_query = np.empty(n_query_peaks, dtype=np.uint8)
    for slot in range(conflict_ids.size):
        reference_id = conflict_ids[slot]
        start, stop = offsets[slot], offsets[slot + 1]
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
            query_peak = query_indices[edge]
            library_peak = library_indices[edge]
            if used_query[query_peak]:
                continue
            seen = False
            for used_index in range(n_used_library):
                if used_library[used_index] == library_peak:
                    seen = True
                    break
            if seen:
                continue
            used_query[query_peak] = 1
            used_library[n_used_library] = library_peak
            n_used_library += 1
            dot_product += products[edge]

        denominator = query_norm * library_norms[reference_id]
        scores[reference_id] = dot_product / denominator if denominator > 0.0 else 0.0
        counts[reference_id] = n_used_library


@njit(cache=True, nogil=True)
def _finalize_candidates(query, product_index, loss_index, library_info, matching, state, touched, n_touched, epoch):
    """Normalize unambiguous pairs, resolve conflicts and apply precursor gate."""
    query_mz, _, precursor_mz, query_norm = query
    reference_precursors, library_norms = library_info
    _, _, _, _, _, _, _, _, _, conflict_slot = state
    scores, counts, conflicts = state[0], state[1], state[2]
    _, _, _, identity_tolerance, identity_ppm = matching

    conflict_ids = np.empty(n_touched, dtype=touched.dtype)
    n_conflicts = 0
    for position in range(n_touched):
        reference_id = touched[position]
        if conflicts[reference_id]:
            conflict_slot[reference_id] = n_conflicts
            conflict_ids[n_conflicts] = reference_id
            n_conflicts += 1
        else:
            denominator = query_norm * library_norms[reference_id]
            scores[reference_id] = scores[reference_id] / denominator if denominator > 0.0 else 0.0

    if n_conflicts:
        conflict_ids = conflict_ids[:n_conflicts]
        offsets = np.empty(n_conflicts + 1, dtype=np.int64)
        offsets[0] = 0
        for slot in range(n_conflicts):
            offsets[slot + 1] = offsets[slot] + counts[conflict_ids[slot]]
        edges = _collect_conflict_edges(
            query,
            product_index,
            loss_index,
            reference_precursors,
            matching,
            state,
            epoch,
            offsets,
        )
        _resolve_compact_conflicts(
            scores,
            counts,
            conflict_ids,
            offsets,
            edges,
            library_norms,
            query_norm,
            query_mz.size,
        )

    if identity_tolerance >= 0.0:
        for position in range(n_touched):
            reference_id = touched[position]
            if not identity_allowed(precursor_mz, reference_precursors[reference_id], identity_tolerance, identity_ppm):
                scores[reference_id] = 0.0
                counts[reference_id] = 0


@njit(cache=True, nogil=True)
def _cosine_topk_row(query, product_index, loss_index, library_info, matching, state, touched, query_bits, epoch):
    """Run candidate discovery and exact assignment for one prepared query.

    Query bundle: ``(m/z, intensity, precursor, L2 norm)``.
    Index bundles: ``(fragment m/z, intensity, owner)``, ``(loss m/z, owner,
    product posting ID)`` and ``(reference precursors, L2 norms)``.
    Matching bundle: ``(mode, tolerance, ppm, identity tolerance, identity ppm)``.
    State bundle holds scores, counts, conflict flags, epoch stamps, cursors,
    compact hybrid bitsets and conflict-slot mappings. No full result row exists.
    """
    _, _, precursor_mz, query_norm = query
    reference_precursors, _ = library_info
    mode, tolerance, use_ppm, _, _ = matching
    touched_count = 0
    if query[0].size == 0 or np.isclose(query_norm, 0.0, rtol=0.0, abs_tol=1e-12):
        return touched, query_bits, touched_count

    if mode != 1:
        touched, query_bits, touched_count = _find_fragment_candidates(
            query,
            product_index,
            state,
            touched,
            query_bits,
            touched_count,
            tolerance,
            use_ppm,
            mode == 2,
            epoch,
        )
    if mode != 0 and np.isfinite(precursor_mz):
        touched, query_bits, touched_count = _find_loss_candidates(
            query,
            product_index,
            loss_index,
            reference_precursors,
            state,
            touched,
            query_bits,
            touched_count,
            tolerance,
            use_ppm,
            mode,
            epoch,
        )
    _finalize_candidates(query, product_index, loss_index, library_info, matching, state, touched, touched_count, epoch)
    return touched, query_bits, touched_count


def _score_cosine_topk_block(
    prepared: PreparedSpectra,
    library: _CosineKernelData,
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
    """Write top-k results for disjoint prepared rows using worker-local buffers.

    Each query accumulates scores and accepted match counts for touched library
    spectra only. The final selection uses the public score dtype, matching
    the dense scorer's output cast before thresholding and tie resolution.
    """
    matching = (mode, tolerance, use_ppm, identity_tol, identity_ppm)
    product_index = (library.peaks_mz, library.peaks_int, library.peaks_spec_idx)
    loss_index = (library.nl_mz, library.nl_spec_idx, library.nl_product_idx)
    library_info = (library.precursor_mz, library.spec_l2)

    for row in range(first_row, last_row):
        first = int(prepared.spec_offsets[row])
        stop = int(prepared.spec_offsets[row + 1])
        if mode == 2:
            workspace.ensure_query_words(stop - first)
        epoch = workspace.next_epoch()
        query = (
            prepared.spec_mz[first:stop],
            prepared.spec_int[first:stop],
            np.float64(prepared.precursor_mz[row]),
            np.float64(prepared.spec_l2[row]),
        )
        state = (
            workspace.scores,
            workspace.candidate_counts,
            workspace.conflicts,
            workspace.epochs,
            workspace.last_query,
            workspace.last_peak,
            workspace.slots,
            workspace.peak_word_epochs,
            workspace.peak_word_bits,
            workspace.conflict_slot,
        )
        touched, query_bits, touched_count = _cosine_topk_row(
            query,
            product_index,
            loss_index,
            library_info,
            matching,
            state,
            workspace.touched,
            workspace.query_bits,
            epoch,
        )
        workspace.touched = touched
        workspace.query_bits = query_bits
        candidates = touched[:touched_count]
        if workspace.selection_scores is not workspace.scores:
            workspace.selection_scores[candidates] = workspace.scores[candidates]
        ids, values = select_topk_candidates(
            candidates,
            workspace.selection_scores,
            k,
            min_score,
        )
        count = ids.size
        indices[row, :count] = ids
        scores[row, :count] = values
        valid_counts[row] = count
        if matches is not None:
            matches[row, :count] = workspace.candidate_counts[ids]
