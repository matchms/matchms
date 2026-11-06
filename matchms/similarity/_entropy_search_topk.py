"""Candidate-aware, exact top-k accumulation for :class:`EntropySearch`.

This module deliberately keeps the existing dense scoring path untouched.
Instead of writing a query-by-library row, each worker owns a library-sized
score buffer and epoch stamps. Only the library IDs reached through inverted
peak postings are recorded and examined during top-k selection.

Short matching windows use Numba, long windows retain the existing bounded
vectorized arithmetic and compiled scatter-add path. For an individual query,
all short windows are processed before all long windows, exactly as in the
dense scorer. Thus float32 additions occur in the same order.
"""

from __future__ import annotations
import numpy as np
from numba import njit
from ._topk import select_topk_candidates


_INITIAL_TOUCHED_CAPACITY = 4096
_MAX_EPOCH = int(np.iinfo(np.uint32).max)


class _TopKWorkspace:
    """Reusable per-worker state, independent of the number of queries.

    Two full-library vectors (scores and epoch stamps) are necessary for cheap
    constant-time candidate updates. The touched-ID vector starts small and
    grows geometrically in Numba only when the query reaches more candidates.
    Nothing is cleared per query except newly encountered score positions.
    """

    def __init__(self, n_references: int, dtype: np.dtype):
        dtype = np.dtype(dtype)
        id_dtype = np.int32 if n_references <= np.iinfo(np.int32).max else np.int64
        self.scores = np.empty(n_references, dtype=dtype)
        self.seen_epochs = np.zeros(n_references, dtype=np.uint32)
        self.touched_ids = np.empty(
            min(n_references, _INITIAL_TOUCHED_CAPACITY),
            dtype=id_dtype,
        )
        self.epoch = 0
        self.mixed: np.ndarray | None = None
        self.contributions: np.ndarray | None = None

    def next_epoch(self) -> int:
        """Advance generation, resetting stamps only on uint32 wraparound."""
        if self.epoch == _MAX_EPOCH:
            self.seen_epochs.fill(0)
            self.epoch = 1
        else:
            self.epoch += 1
        return self.epoch

    def ensure_vector_capacity(self, capacity: int, dtype: np.dtype) -> None:
        """Allocate at most the configured vector-chunk length per worker."""
        if self.mixed is None or self.mixed.size < capacity:
            self.mixed = np.empty(capacity, dtype=dtype)
            self.contributions = np.empty_like(self.mixed)


@njit(cache=True, inline="always")
def _add_candidate(scores, epochs, touched, n_touched, epoch, library_id, delta):
    """Accumulate a posting; append its library ID at most once per query."""
    if epochs[library_id] != epoch:
        if n_touched == touched.size:
            # Geometric growth avoids an N-library-sized touched list when a
            # query matches only a small subset of references.
            new_size = min(epochs.size, max(1, touched.size * 2))
            grown = np.empty(new_size, dtype=touched.dtype)
            for p in range(n_touched):
                grown[p] = touched[p]
            touched = grown
        epochs[library_id] = epoch
        scores[library_id] = 0.0
        touched[n_touched] = library_id
        n_touched += 1
    scores[library_id] += delta
    return touched, n_touched


@njit(cache=True, nogil=True)
def _accumulate_short_topk(
    query_intensity,
    library_intensity,
    library_terms,
    library_ids,
    bounds,
    first_peak,
    last_peak,
    min_vector_hits,
    scores,
    epochs,
    touched,
    epoch,
):
    """Visit short windows in precisely the dense scorer's peak order."""
    n_touched = 0
    for peak in range(first_peak, last_peak):
        low, high = bounds[peak]
        if high - low >= min_vector_hits:
            continue
        value = query_intensity[peak]
        query_term = value * np.log2(value)
        for posting in range(low, high):
            mixed = value + library_intensity[posting]
            delta = mixed * np.log2(mixed) - library_terms[posting] - query_term
            touched, n_touched = _add_candidate(
                scores,
                epochs,
                touched,
                n_touched,
                epoch,
                library_ids[posting],
                delta,
            )
    return touched, n_touched


@njit(cache=True, nogil=True)
def _scatter_add_topk(scores, epochs, touched, n_touched, epoch, library_ids, deltas):
    """Scatter a bounded vector chunk, retaining all encountered library IDs."""
    for p in range(library_ids.size):
        touched, n_touched = _add_candidate(
            scores,
            epochs,
            touched,
            n_touched,
            epoch,
            library_ids[p],
            deltas[p],
        )
    return touched, n_touched


def _score_topk_block(
    prepared,
    view,
    bounds,
    workspace: _TopKWorkspace,
    indices: np.ndarray,
    scores: np.ndarray,
    valid_counts: np.ndarray,
    first_row: int,
    last_row: int,
    k: int,
    min_score: float,
    vector_min_hits: int,
    vector_chunk_size: int,
) -> None:
    """Write compact results for disjoint query rows using worker-local state.

    This function never allocates a dense query-by-library score array. Each
    candidate's full score is accumulated before selecting the best ``k``.
    Ranking reads only ``touched_ids[:n_touched]``.
    """
    offsets = prepared.spec_offsets
    query_intensity = prepared.spec_int
    for row in range(first_row, last_row):
        epoch = workspace.next_epoch()
        first_peak, last_peak = int(offsets[row]), int(offsets[row + 1])

        touched, n_touched = _accumulate_short_topk(
            query_intensity,
            view.intensity,
            view.terms,
            view.spectrum_ids,
            bounds,
            first_peak,
            last_peak,
            vector_min_hits,
            workspace.scores,
            workspace.seen_epochs,
            workspace.touched_ids,
            epoch,
        )

        # The original dense path scores *all short* windows first, then *all
        # long* windows. That ordering is preservedeven when the matching windows
        # overlap and float32 additions are not exactly associative.
        for peak in range(first_peak, last_peak):
            low, high = bounds[peak]
            width = int(high - low)
            if width < vector_min_hits:
                continue
            workspace.ensure_vector_capacity(
                min(width, vector_chunk_size),
                view.intensity.dtype,
            )
            capacity = workspace.mixed.size
            value = query_intensity[peak]
            query_term = value * np.log2(value)
            for start in range(int(low), int(high), capacity):
                stop = min(start + capacity, int(high))
                mixture = workspace.mixed[: stop - start]
                delta = workspace.contributions[: stop - start]
                np.add(view.intensity[start:stop], value, out=mixture)
                np.log2(mixture, out=delta)
                np.multiply(mixture, delta, out=delta)
                np.subtract(delta, view.terms[start:stop], out=delta)
                np.subtract(delta, query_term, out=delta)
                touched, n_touched = _scatter_add_topk(
                    workspace.scores,
                    workspace.seen_epochs,
                    touched,
                    n_touched,
                    epoch,
                    view.spectrum_ids[start:stop],
                    delta,
                )

        workspace.touched_ids = touched
        ids, values = select_topk_candidates(
            touched[:n_touched],
            workspace.scores,
            k,
            min_score,
        )
        n_hits = ids.size
        indices[row, :n_hits] = ids
        scores[row, :n_hits] = values
        valid_counts[row] = n_hits
