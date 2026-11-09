"""Exact touched-candidate top-k retrieval for EntropyFlash and Entropy.

The dense entropy kernel is intentionally left unchanged. This kernel visits the
same ordered postings and performs the same float64 contribution arithmetic,
but never creates an output row of length ``n_references``. Each worker owns a
reusable scratch state where only touched library IDs are initialized and ranked.

Hybrid matching needs two kinds of cross-pass state. Physical product peaks are
tracked with epoch-stamped, 64-peak bitset *words*, not an int64 epoch per peak.
Matched query peaks are tracked in a dynamically grown bitmap indexed by the
slot of each touched library spectrum, not by every library spectrum.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from ._flash_entropy import xlog2
from ._flash_tolerance import identity_allowed, window, within
from ._flash_topk_common import (
    _FlashTopKWorkspace,
    _register_touched,
)
from ._topk import select_topk_candidates


class _EntropyTopKWorkspace(_FlashTopKWorkspace):
    """Entropy cursors layered over the common epoch/compact-bitset workspace."""

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        super().__init__(n_references, n_peaks, dtype, hybrid)
        self.last_peak = np.empty(n_references, dtype=np.int64)
        self.last_query = np.empty(n_references, dtype=np.int64)


@njit(cache=True, inline="always")
def _touch(scores, epochs, last_peak, last_query, slots, touched, query_bits, n_touched, epoch, column, initial_peak):
    """Register an entropy candidate and initialize its one-to-one cursors."""
    touched, query_bits, n_touched, is_new = _register_touched(
        epochs,
        slots,
        touched,
        query_bits,
        n_touched,
        epoch,
        column,
    )
    if is_new:
        scores[column] = 0.0
        last_peak[column] = initial_peak
        last_query[column] = -1
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _entropy_topk_row(
    q_mz,
    q_int,
    q_precursor,
    fragment,
    neutral_loss,
    lib_pmz,
    n_lib_peaks,
    tolerance,
    use_ppm,
    mode,
    identity_tol,
    identity_ppm,
    scores,
    epochs,
    last_peak,
    last_query,
    slots,
    touched,
    query_bits,
    peak_word_epochs,
    peak_word_bits,
    epoch,
):
    """Accumulate a complete row and returns touched IDs and dynamically grown buffers.

    This duplicates the dense kernel's posting traversal and acceptance order,
    including fragment-first hybrid rules and last-peak monotonicity. Only
    bookkeeping and final score materialization differ.
    """
    fragment_mz, fragment_int, fragment_terms, fragment_spec, fragment_pid = fragment
    loss_mz, loss_int, loss_terms, loss_spec, loss_pid = neutral_loss
    n_touched = 0
    has_precursor = np.isfinite(q_precursor)

    if mode != 1:
        for query_index in range(q_mz.size):
            coordinate = np.float64(q_mz[query_index])
            intensity = np.float64(q_int[query_index])
            if intensity <= 0.0:
                continue
            query_term = xlog2(intensity)
            left, right = window(fragment_mz, coordinate, tolerance, use_ppm)
            for posting in range(left, right):
                column = fragment_spec[posting]
                peak_id = fragment_pid[posting]
                if epochs[column] == epoch:
                    if last_query[column] == query_index or peak_id <= last_peak[column]:
                        continue
                library_intensity = np.float64(fragment_int[posting])
                if library_intensity <= 0.0:
                    continue
                if not within(coordinate, np.float64(fragment_mz[posting]), tolerance, use_ppm):
                    continue
                touched, query_bits, n_touched = _touch(
                    scores,
                    epochs,
                    last_peak,
                    last_query,
                    slots,
                    touched,
                    query_bits,
                    n_touched,
                    epoch,
                    column,
                    -1,
                )
                scores[column] += xlog2(intensity + library_intensity) - query_term - fragment_terms[posting]
                last_query[column] = query_index
                last_peak[column] = peak_id
                if mode == 2:
                    word = peak_id // 64
                    if peak_word_epochs[word] != epoch:
                        peak_word_epochs[word] = epoch
                        peak_word_bits[word] = np.uint64(0)
                    peak_word_bits[word] |= np.uint64(1) << np.uint64(peak_id % 64)
                    query_bits[slots[column], query_index // 64] |= np.uint64(1) << np.uint64(query_index % 64)

    if mode != 0 and has_precursor:
        # The loss pass starts its peak and query cursors afresh. Only references
        # actually reached by the fragment pass need resetting.
        for position in range(n_touched):
            column = touched[position]
            last_peak[column] = n_lib_peaks
            last_query[column] = -1

        for query_index in range(q_mz.size - 1, -1, -1):
            intensity = np.float64(q_int[query_index])
            if intensity <= 0.0:
                continue
            coordinate = np.float64(q_precursor) - np.float64(q_mz[query_index])
            query_term = xlog2(intensity)
            left, right = window(loss_mz, coordinate, tolerance, use_ppm)
            for posting in range(left, right):
                column = loss_spec[posting]
                peak_id = loss_pid[posting]
                if epochs[column] == epoch:
                    if last_query[column] == query_index or peak_id >= last_peak[column]:
                        continue
                if mode == 2:
                    word = peak_id // 64
                    if peak_word_epochs[word] == epoch and peak_word_bits[word] & (
                        np.uint64(1) << np.uint64(peak_id % 64)
                    ):
                        continue
                    if epochs[column] == epoch and query_bits[slots[column], query_index // 64] & (
                        np.uint64(1) << np.uint64(query_index % 64)
                    ):
                        continue
                library_intensity = np.float64(loss_int[posting])
                if library_intensity <= 0.0:
                    continue
                if not within(coordinate, loss_mz[posting], tolerance, use_ppm):
                    continue
                touched, query_bits, n_touched = _touch(
                    scores,
                    epochs,
                    last_peak,
                    last_query,
                    slots,
                    touched,
                    query_bits,
                    n_touched,
                    epoch,
                    column,
                    n_lib_peaks,
                )
                scores[column] += xlog2(intensity + library_intensity) - query_term - loss_terms[posting]
                last_query[column] = query_index
                last_peak[column] = peak_id

    if identity_tol >= 0.0:
        for position in range(n_touched):
            column = touched[position]
            if not identity_allowed(q_precursor, lib_pmz[column], identity_tol, identity_ppm):
                scores[column] = 0.0
    return touched, query_bits, n_touched


def _score_entropy_topk_block(
    prepared,
    library,
    workspace: _EntropyTopKWorkspace,
    indices: np.ndarray,
    scores: np.ndarray,
    valid_counts: np.ndarray,
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
    """Write independent compact rows, reusing only this worker's scratch arrays."""
    for row in range(first_row, last_row):
        first = int(prepared.spec_offsets[row])
        last = int(prepared.spec_offsets[row + 1])
        if mode == 2:
            workspace.ensure_query_words(last - first)
        epoch = workspace.next_epoch()
        touched, query_bits, n_touched = _entropy_topk_row(
            prepared.spec_mz[first:last],
            prepared.spec_int[first:last],
            np.float64(prepared.precursor_mz[row]),
            library.fragment,
            library.neutral_loss,
            library.precursor_mz,
            library.n_peaks,
            tolerance,
            use_ppm,
            mode,
            identity_tol,
            identity_ppm,
            workspace.scores,
            workspace.epochs,
            workspace.last_peak,
            workspace.last_query,
            workspace.slots,
            workspace.touched,
            workspace.query_bits,
            workspace.peak_word_epochs,
            workspace.peak_word_bits,
            epoch,
        )
        workspace.touched = touched
        workspace.query_bits = query_bits
        candidates = touched[:n_touched]
        if workspace.selection_scores is not workspace.scores:
            # Dense entropy accumulates in float64 and casts once, on output.
            # Selection (including ties and cutoff) must use that cast dtype.
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
