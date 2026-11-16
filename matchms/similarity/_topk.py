"""Deterministic partial selection for exact top-k similarity retrieval.

Both dense-row and touched-candidate selectors rank by descending
score, then ascending reference ID. They never sort all library spectra.
"""

from __future__ import annotations
import math
import operator
import numpy as np


def validate_topk_parameters(
    k: int,
    min_score: float,
    batch_size: int | None,
) -> tuple[int, float, int | None]:
    """Validate retrieval parameters without silently rounding integer inputs."""
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be a positive integer, not bool.")
    k = operator.index(k)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if isinstance(min_score, (bool, np.bool_)):
        raise TypeError("min_score must be a finite nonnegative number, not bool.")
    min_score = float(min_score)
    if not math.isfinite(min_score) or min_score < 0:
        raise ValueError("min_score must be finite and nonnegative.")

    if batch_size is not None:
        if isinstance(batch_size, (bool, np.bool_)):
            raise TypeError("batch_size must be a positive integer, not bool.")
        batch_size = operator.index(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

    return k, min_score, batch_size


def select_topk_row(
    row: np.ndarray,
    k: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact top-k positive hits from one already-computed dense row.

    Scores must be strictly greater than ``min_score``. Rankings use descending
    score followed by ascending library ID. ``np.partition`` determines the kth
    score threshold in expected O(n) time; choosing the smallest IDs among *all*
    entries tied at that threshold makes the result deterministic, unlike the
    arbitrary boundary tie selection of a bare ``np.argpartition`` call.

    Only the selected hits are fully sorted. Returned score values retain the
    dtype and numerical values of the input row (float32 or float64).
    """
    candidates = np.flatnonzero(row > min_score)
    if candidates.size == 0:
        return candidates, np.empty(0, dtype=row.dtype)

    if candidates.size > k:
        candidate_scores = row[candidates]
        threshold = np.partition(candidate_scores, candidate_scores.size - k)[candidate_scores.size - k]
        above = candidates[candidate_scores > threshold]
        # candidates are already in ascending ID order, so selecting the first
        # tied IDs implements the documented secondary ranking key.
        tied = candidates[candidate_scores == threshold][: k - above.size]
        selected = np.concatenate((above, tied))
    else:
        selected = candidates

    selected_scores = row[selected]
    order = np.lexsort((selected, -selected_scores))
    return selected[order], selected_scores[order]


def select_topk_candidates(
    touched_ids: np.ndarray,
    score_buffer: np.ndarray,
    k: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact top-k hits from the *touched* subset of an indexed library.

    ``touched_ids`` contains unique reference IDs in arbitrary encounter order;
    its size may be much smaller than ``score_buffer.size``. This function never
    scans the complete score buffer. Only scores > ``min_score`` qualify.

    The kth-score boundary is handled explicitly: ties are resolved by choosing
    the smallest library IDs, independent of posting encounter order. A full
    ordering is applied only to the final ``k`` hits, not to every candidate.
    Values remain in the scorer's output dtype, preserving dense-row ranking.
    """
    # Gather just the touched subset, not the N-library score buffer.
    values = score_buffer[touched_ids]
    qualified = values > min_score
    ids = touched_ids[qualified]
    values = values[qualified]
    if ids.size == 0:
        return ids, values

    if ids.size > k:
        boundary = np.partition(values, values.size - k)[values.size - k]
        above = values > boundary
        high_ids = ids[above]
        high_scores = values[above]
        remaining = k - high_ids.size
        tied_ids = ids[values == boundary]
        if tied_ids.size > remaining:
            tied_ids = np.partition(tied_ids, remaining - 1)[:remaining]
        ids = np.concatenate((high_ids, tied_ids))
        values = np.concatenate(
            (
                high_scores,
                np.full(tied_ids.size, boundary, dtype=values.dtype),
            )
        )

    order = np.lexsort((ids, -values))
    return ids[order], values[order]
