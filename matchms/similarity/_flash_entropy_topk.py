"""Exact, candidate-aware EntropyFlash retrieval without dense result rows.

The score is accumulated in the same float64 order as the dense entropy
kernel. A worker initializes state *only when a reference is encountered*,
then selects the best k references among those touched IDs. Fragment-first
hybrid matching uses compact query bitmaps and epoch-stamped physical peaks.

The two posting sweeps are separate so their different cursor directions and
cross-pass matching rules can be reviewed independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from ._flash_entropy import xlog2
from ._flash_tolerance import identity_allowed, window, within
from ._flash_topk_common import _MAX_EPOCH, _FlashTopKWorkspace, _register_touched
from ._topk import select_topk_candidates

if TYPE_CHECKING:
    from ._flash_prepared import PreparedSpectra
    from .flash_index import _EntropyKernelData


class _EntropyTopKWorkspace(_FlashTopKWorkspace):
    """Worker-owned touched-library state and entropy one-to-one cursors.

    ``last_peak`` and ``last_query`` are initialized lazily on a spectrum's
    first encounter in a query epoch. Their contents for untouched reference
    IDs are undefined and must not be inspected.
    """

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        super().__init__(n_references, n_peaks, dtype, hybrid)
        self.last_peak = np.empty(n_references, dtype=np.int64)
        self.last_query = np.empty(n_references, dtype=np.int64)


@njit(cache=True, inline="always")
def _touch_entropy(state, touched, query_bits, n_touched, epoch, reference_id, initial_peak):
    """Register one reference and initialize its entropy cursors if new."""
    scores, epochs, last_peak, last_query, slots, _, _ = state
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
        last_peak[reference_id] = initial_peak
        last_query[reference_id] = -1
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _match_fragment_postings(query, fragment, state, touched, query_bits, n_touched, tolerance, use_ppm, hybrid, epoch):
    """Accumulate direct matches in ascending fragment and physical-peak order."""
    query_mz, query_intensity, _ = query
    mz, intensity_values, entropy_terms, owners, physical_ids = fragment
    scores, epochs, last_peak, last_query, slots, peak_word_epochs, peak_word_bits = state

    for query_peak in range(query_mz.size):
        coordinate = np.float64(query_mz[query_peak])
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        query_term = xlog2(intensity)
        first, stop = window(mz, coordinate, tolerance, use_ppm)
        for posting in range(first, stop):
            reference_id = owners[posting]
            physical_peak = physical_ids[posting]
            if epochs[reference_id] == epoch:
                if last_query[reference_id] == query_peak or physical_peak <= last_peak[reference_id]:
                    continue
            reference_intensity = np.float64(intensity_values[posting])
            if reference_intensity <= 0.0:
                continue
            if not within(coordinate, np.float64(mz[posting]), tolerance, use_ppm):
                continue
            touched, query_bits, n_touched = _touch_entropy(
                state,
                touched,
                query_bits,
                n_touched,
                epoch,
                reference_id,
                -1,
            )
            scores[reference_id] += xlog2(intensity + reference_intensity) - query_term - entropy_terms[posting]
            last_query[reference_id] = query_peak
            last_peak[reference_id] = physical_peak
            if hybrid:
                word = physical_peak // 64
                if peak_word_epochs[word] != epoch:
                    peak_word_epochs[word] = epoch
                    peak_word_bits[word] = np.uint64(0)
                peak_word_bits[word] |= np.uint64(1) << np.uint64(physical_peak % 64)
                query_bits[slots[reference_id], query_peak // 64] |= np.uint64(1) << np.uint64(query_peak % 64)
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _match_neutral_loss_postings(
    query, neutral_loss, state, touched, query_bits, n_touched, n_library_peaks, tolerance, use_ppm, hybrid, epoch
):
    """Accumulate losses in ascending loss order, without reusing hybrid peaks."""
    query_mz, query_intensity, precursor_mz = query
    mz, intensity_values, entropy_terms, owners, physical_ids = neutral_loss
    scores, epochs, last_peak, last_query, slots, peak_word_epochs, peak_word_bits = state

    # The neutral-loss sweep uses a reversed physical-peak cursor. Reset only
    # references encountered during this query's fragment sweep.
    for position in range(n_touched):
        reference_id = touched[position]
        last_peak[reference_id] = n_library_peaks
        last_query[reference_id] = -1

    for query_peak in range(query_mz.size - 1, -1, -1):
        intensity = np.float64(query_intensity[query_peak])
        if intensity <= 0.0:
            continue
        coordinate = np.float64(precursor_mz) - np.float64(query_mz[query_peak])
        query_term = xlog2(intensity)
        first, stop = window(mz, coordinate, tolerance, use_ppm)
        for posting in range(first, stop):
            reference_id = owners[posting]
            physical_peak = physical_ids[posting]
            if epochs[reference_id] == epoch:
                if last_query[reference_id] == query_peak or physical_peak >= last_peak[reference_id]:
                    continue
            if hybrid:
                word = physical_peak // 64
                if peak_word_epochs[word] == epoch and peak_word_bits[word] & (
                    np.uint64(1) << np.uint64(physical_peak % 64)
                ):
                    continue
                if epochs[reference_id] == epoch and query_bits[slots[reference_id], query_peak // 64] & (
                    np.uint64(1) << np.uint64(query_peak % 64)
                ):
                    continue
            reference_intensity = np.float64(intensity_values[posting])
            if reference_intensity <= 0.0:
                continue
            if not within(coordinate, mz[posting], tolerance, use_ppm):
                continue
            touched, query_bits, n_touched = _touch_entropy(
                state,
                touched,
                query_bits,
                n_touched,
                epoch,
                reference_id,
                n_library_peaks,
            )
            scores[reference_id] += xlog2(intensity + reference_intensity) - query_term - entropy_terms[posting]
            last_query[reference_id] = query_peak
            last_peak[reference_id] = physical_peak
    return touched, query_bits, n_touched


@njit(cache=True, nogil=True)
def _apply_identity_to_touched(query_precursor, reference_precursors, state, touched, n_touched, tolerance, use_ppm):
    """Zero candidates outside the optional precursor-identity gate."""
    scores = state[0]
    for position in range(n_touched):
        reference_id = touched[position]
        if not identity_allowed(query_precursor, reference_precursors[reference_id], tolerance, use_ppm):
            scores[reference_id] = 0.0


@njit(cache=True, nogil=True)
def _entropy_topk_row(query, fragment, neutral_loss, library_info, settings, state, touched, query_bits, epoch):
    """Execute both entropy matching passes and return the touched references.

    ``query`` contains m/z, intensity and precursor; ``library_info`` contains
    precursor m/z and the number of physical peaks. ``settings`` holds matching
    mode, peak tolerance/ppm and optional identity-gate tolerance/ppm.
    ``state`` groups worker-owned score, epoch, cursors and hybrid bitset arrays.
    """
    _, _, precursor_mz = query
    library_precursors, n_library_peaks = library_info
    mode, tolerance, use_ppm, identity_tol, identity_ppm = settings
    hybrid = mode == 2
    touched_count = 0
    if mode != 1:
        touched, query_bits, touched_count = _match_fragment_postings(
            query,
            fragment,
            state,
            touched,
            query_bits,
            touched_count,
            tolerance,
            use_ppm,
            hybrid,
            epoch,
        )
    if mode != 0 and np.isfinite(precursor_mz):
        touched, query_bits, touched_count = _match_neutral_loss_postings(
            query,
            neutral_loss,
            state,
            touched,
            query_bits,
            touched_count,
            n_library_peaks,
            tolerance,
            use_ppm,
            hybrid,
            epoch,
        )
    if identity_tol >= 0.0:
        _apply_identity_to_touched(
            precursor_mz, library_precursors, state, touched, touched_count, identity_tol, identity_ppm
        )
    return touched, query_bits, touched_count


def _score_entropy_topk_block(
    prepared: PreparedSpectra,
    library: _EntropyKernelData,
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
    """Score independent query rows using reusable worker-local scratch state.

    ``prepared`` is a read-only packed query batch; ``library`` is its validated
    entropy index view. Output arrays are preallocated and rows are disjoint
    between worker calls. Each row writes at most ``k`` indices and scores.
    """
    settings = (mode, tolerance, use_ppm, identity_tol, identity_ppm)
    library_info = (library.precursor_mz, library.n_peaks)
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
        )
        state = (
            workspace.scores,
            workspace.epochs,
            workspace.last_peak,
            workspace.last_query,
            workspace.slots,
            workspace.peak_word_epochs,
            workspace.peak_word_bits,
        )
        touched, query_bits, touched_count = _entropy_topk_row(
            query,
            library.fragment,
            library.neutral_loss,
            library_info,
            settings,
            state,
            workspace.touched,
            workspace.query_bits,
            epoch,
        )
        workspace.touched = touched
        workspace.query_bits = query_bits
        candidates = touched[:touched_count]
        if workspace.selection_scores is not workspace.scores:
            # The dense scorer also casts its float64 accumulations only once.
            workspace.selection_scores[candidates] = workspace.scores[candidates]
        ids, values = select_topk_candidates(candidates, workspace.selection_scores, k, min_score)
        count = ids.size
        indices[row, :count] = ids
        scores[row, :count] = values
        valid_counts[row] = count
