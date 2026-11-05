"""Selection and bounded-workspace utilities for exact top-k similarity search.

These helpers operate on already-computed dense score rows.
"""

from __future__ import annotations
import math
import operator
import numpy as np


# Only the dense temporary scoring batch is bounded by this target. A single
# library-sized score row can exceed this budget for exceptionally large indexes.
_DEFAULT_BATCH_BYTES = 64 * 1024 * 1024
_MAX_AUTO_BATCH_ROWS = 32


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


def resolve_batch_size(
    batch_size: int | None,
    n_queries: int,
    n_references: int,
    itemsize: int,
) -> int:
    """Choose a query batch bounded by the 64-MiB default score-buffer budget.

    An explicit batch size overrides the automatic target. The calculation only
    controls the dense score buffer, not index storage or kernel scratch memory.
    """
    if batch_size is None:
        bytes_per_row = max(1, n_references * itemsize)
        batch_size = max(1, min(_MAX_AUTO_BATCH_ROWS, _DEFAULT_BATCH_BYTES // bytes_per_row))
    return min(max(1, n_queries), batch_size)


def select_topk_row(
    row: np.ndarray,
    k: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact top-k positive hits from one already-computed score row.

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
