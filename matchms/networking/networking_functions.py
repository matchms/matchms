"""Helper functions to build and handle spectral networks."""
from typing import Optional, Sequence, Tuple
import numpy as np
from matchms import Scores


def get_top_hits(
    scores: Scores,
    top_n: int = 25,
    axis: int = 1,
    score_name: Optional[str] = None,
    identifiers: Optional[Sequence] = None,
    ignore_diagonal: bool = False,
) -> Tuple[dict, dict]:
    """Get top_n highest scores and corresponding indices for each row or column.

    Parameters
    ----------
    scores
        Matchms Scores object containing similarity values.
    top_n
        Number of top hits to return per row or column.
    axis
        Axis along which to search:
        - ``axis=1``: get top hits for each row
        - ``axis=0``: get top hits for each column
    score_name
        Name of the score field to use when ``scores`` contains multiple fields.
        If None:
        - scalar Scores: the only field is used
        - multi-field Scores: defaults to ``"score"`` if available, otherwise raises
    identifiers
        Optional identifiers for the selected axis.
        - for ``axis=1``, must have length ``scores.shape[0]``
        - for ``axis=0``, must have length ``scores.shape[1]``
        If None, integer indices are used as dictionary keys.
    ignore_diagonal
        If True, diagonal self-hits are excluded. This is only meaningful for
        square score matrices where row and column indices refer to the same set.

    Returns
    -------
    similars_idx, similars_scores
        Two dictionaries:
        - keys are identifiers (or integer row/column indices)
        - values are NumPy arrays of hit indices and hit scores
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    matrix = _get_score_array(scores, score_name)
    n_rows, n_cols = matrix.shape

    expected_len = n_rows if axis == 1 else n_cols
    if identifiers is None:
        identifiers = list(range(expected_len))
    else:
        if len(identifiers) != expected_len:
            raise ValueError(
                f"identifiers must have length {expected_len} for axis={axis}, "
                f"but got length {len(identifiers)}."
            )

    if ignore_diagonal and n_rows != n_cols:
        raise ValueError("ignore_diagonal=True requires a square score matrix.")

    if axis == 1:
        return _get_top_hits_along_rows(matrix, identifiers, top_n, ignore_diagonal)
    return _get_top_hits_along_columns(matrix, identifiers, top_n, ignore_diagonal)


def get_top_hits_by_row(
    scores: Scores,
    top_n: int = 25,
    score_name: Optional[str] = None,
    identifiers: Optional[Sequence] = None,
    ignore_diagonal: bool = False,
) -> Tuple[dict, dict]:
    """Get top hits for each row."""
    return get_top_hits(
        scores=scores,
        top_n=top_n,
        axis=1,
        score_name=score_name,
        identifiers=identifiers,
        ignore_diagonal=ignore_diagonal,
    )


def get_top_hits_by_column(
    scores: Scores,
    top_n: int = 25,
    score_name: Optional[str] = None,
    identifiers: Optional[Sequence] = None,
    ignore_diagonal: bool = False,
) -> Tuple[dict, dict]:
    """Get top hits for each column."""
    return get_top_hits(
        scores=scores,
        top_n=top_n,
        axis=0,
        score_name=score_name,
        identifiers=identifiers,
        ignore_diagonal=ignore_diagonal,
    )


def _get_score_array(scores: Scores, score_name: Optional[str]) -> np.ndarray:
    """Return the selected score field as a dense NumPy array."""
    if score_name is None:
        if scores.is_scalar:
            return scores.to_array()
        if "score" in scores.score_fields:
            return scores["score"].to_array()
        raise KeyError(
            "score_name must be provided for multi-field Scores when no 'score' field exists. "
            f"Available fields: {scores.score_fields}."
        )

    return scores[score_name].to_array()


def _sorted_top_indices(
    values: np.ndarray,
    top_n: int,
    exclude_index: Optional[int] = None,
) -> np.ndarray:
    """Return top indices sorted by descending score, ties by ascending index."""
    if top_n <= 0 or len(values) == 0:
        return np.array([], dtype=int)

    extra = 1 if exclude_index is not None else 0
    n_select = min(top_n + extra, len(values))

    candidate_idx = np.argpartition(values, -n_select)[-n_select:]

    if exclude_index is not None:
        candidate_idx = candidate_idx[candidate_idx != exclude_index]

    # Sort by descending score, then ascending index
    candidate_scores = values[candidate_idx]
    order = np.lexsort((candidate_idx, -candidate_scores))
    return candidate_idx[order][:top_n]


def _get_top_hits_along_rows(
    matrix: np.ndarray,
    identifiers: Sequence,
    top_n: int,
    ignore_diagonal: bool,
) -> Tuple[dict, dict]:
    """Get top hits for each row."""
    similars_idx = {}
    similars_scores = {}

    for i in range(matrix.shape[0]):
        values = matrix[i, :]
        order = _sorted_top_indices(
            values,
            top_n=top_n,
            exclude_index=i if ignore_diagonal else None,
        )
        similars_idx[identifiers[i]] = order
        similars_scores[identifiers[i]] = values[order]

    return similars_idx, similars_scores


def _get_top_hits_along_columns(
    matrix: np.ndarray,
    identifiers: Sequence,
    top_n: int,
    ignore_diagonal: bool,
) -> Tuple[dict, dict]:
    """Get top hits for each column."""
    similars_idx = {}
    similars_scores = {}

    for j in range(matrix.shape[1]):
        values = matrix[:, j]
        order = _sorted_top_indices(
            values,
            top_n=top_n,
            exclude_index=j if ignore_diagonal else None,
        )
        similars_idx[identifiers[j]] = order
        similars_scores[identifiers[j]] = values[order]

    return similars_idx, similars_scores
