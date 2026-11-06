"""Exact touched-candidate top-k retrieval for EntropyFlash and Entropy.

The dense entropy kernel is intentionally left unchanged. This kernel visits the
same ordered postings and performs the same float64 contribution arithmetic,
but never creates an output row of length ``n_references``. Each worker owns
reusable scratch state; only touched library IDs are initialized and ranked.

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
from ._topk import select_topk_candidates


_INITIAL_CAPACITY = 256
_MAX_EPOCH = int(np.iinfo(np.uint32).max)


class _EntropyTopKWorkspace:
    """Scratch allocations owned by exactly one search worker thread.

    ``scores`` uses float64 to reproduce the existing entropy kernel's
    accumulation, regardless of the public score dtype. ``selection_scores``
    converts touched entries to that dtype only once their scores are complete.

    ``epochs`` eliminates the O(n_references) zero fill per query. ``touched``
    and the hybrid query bitmap grow only when new library IDs are encountered.
    """

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        dtype = np.dtype(dtype)
        index_dtype = np.int32 if n_references <= np.iinfo(np.int32).max else np.int64
        self.scores = np.empty(n_references, dtype=np.float64)
        self.selection_scores = self.scores if dtype == np.dtype(np.float64) else np.empty(n_references, dtype=dtype)
        self.epochs = np.zeros(n_references, dtype=np.uint32)
        self.last_peak = np.empty(n_references, dtype=np.int64)
        self.last_query = np.empty(n_references, dtype=np.int64)
        capacity = min(n_references, _INITIAL_CAPACITY)
        self.touched = np.empty(capacity, dtype=index_dtype)
        self.slots = np.empty(n_references if hybrid else 0, dtype=index_dtype)
        self.query_bits = np.empty((capacity, 0), dtype=np.uint64)
        n_peak_words = (n_peaks + 63) // 64 if hybrid else 0
        self.peak_word_epochs = np.zeros(n_peak_words, dtype=np.uint32)
        self.peak_word_bits = np.empty(n_peak_words, dtype=np.uint64)
        self.epoch = 0

    def next_epoch(self) -> int:
        """Advance generation, resetting stamps only at uint32 wraparound."""
        if self.epoch == _MAX_EPOCH:
            self.epochs.fill(0)
            self.peak_word_epochs.fill(0)
            self.epoch = 1
        else:
            self.epoch += 1
        return self.epoch

    def ensure_query_words(self, n_query_peaks: int) -> None:
        """Grow hybrid query bitmap width between rows, never during a row."""
        words = (n_query_peaks + 63) // 64
        if words > self.query_bits.shape[1]:
            # All rows used by this query will be reset lazily by _touch.
            self.query_bits = np.empty((self.touched.size, words), dtype=np.uint64)


@njit(cache=True, inline="always")
def _touch(scores, epochs, last_peak, last_query, slots, touched, query_bits, n_touched, epoch, column, initial_peak):
    """Register a newly matched spectrum, resizing touched state if necessary."""
    if epochs[column] != epoch:
        if n_touched == touched.size:
            new_size = min(epochs.size, max(1, touched.size * 2))
            grown = np.empty(new_size, dtype=touched.dtype)
            grown[:n_touched] = touched[:n_touched]
            touched = grown
            if query_bits.shape[1]:
                grown_bits = np.empty((new_size, query_bits.shape[1]), dtype=np.uint64)
                grown_bits[:n_touched, :] = query_bits[:n_touched, :]
                query_bits = grown_bits
            else:
                query_bits = np.empty((new_size, 0), dtype=np.uint64)
        epochs[column] = epoch
        scores[column] = 0.0
        last_peak[column] = initial_peak
        last_query[column] = -1
        touched[n_touched] = column
        if query_bits.shape[1]:
            slots[column] = n_touched
            query_bits[n_touched, :] = 0
        n_touched += 1
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
    """Accumulate a complete row; return touched IDs and dynamically grown buffers.

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
