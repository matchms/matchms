import logging
from typing import Iterator, Tuple, Union
import numpy as np
from scipy.sparse import coo_array


logger = logging.getLogger("matchms")


class ScoresMask:
    """Row and column indexes for a COO matrix

    Can be used as a mask to specify between which pairs similarity scores should be computed"""

    def __init__(self, idx_row: np.ndarray, idx_col: np.ndarray):
        self._idx_row = idx_row
        self._idx_col = idx_col
        if not (self._idx_row.ndim == 1 and self._idx_col.ndim == 1):
            raise ValueError("idx_row and idx_col must be 1D")
        if self._idx_row.shape[0] != self._idx_col.shape[0]:
            raise ValueError("idx_row and idx_col must have same length")
        self.length = self._idx_row.shape[0]

    @classmethod
    def from_matrix(cls, scores_matrix: np.ndarray, operation: str, value) -> "ScoresMask":
        operator = _get_operator(operation)
        mask = operator(scores_matrix, value)
        rows, cols = np.where(mask)
        return cls(rows, cols)

    @classmethod
    def from_coo_array(cls, scores_coo_array: coo_array, operation: str, value) -> "ScoresMask":
        operator = _get_operator(operation)
        if operator(np.float64(0.0), value):
            logger.warning(
                "The condition '%s %s' would include 0.0 values, but COO arrays may not store "
                "explicit zeros. Scores of exactly 0.0 may be missing from the result. "
                "Use ScoresMask.from_matrix() to avoid this issue.",
                operation,
                value,
            )
        mask = operator(scores_coo_array.data, value)
        rows = scores_coo_array.row[mask]
        cols = scores_coo_array.col[mask]
        return cls(rows, cols)

    @property
    def idx_row(self) -> np.ndarray:
        return self._idx_row

    @property
    def idx_col(self) -> np.ndarray:
        return self._idx_col

    def __iter__(self) -> Iterator[Tuple[np.intp, np.intp]]:
        return zip(self._idx_row, self._idx_col)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: Union[int, slice]) -> Union[Tuple[np.ndarray, np.ndarray], "ScoresMask"]:
        if isinstance(item, int):
            return self._idx_row[item], self._idx_col[item]
        if isinstance(item, slice):
            return ScoresMask(self._idx_row[item], self._idx_col[item])  # Return a COOIndex subset
        raise TypeError("Index must be an integer or a slice")


def _get_operator(relation: str):
    """
    Retrieve the numpy comparison function corresponding to the provided operator string.

    Parameters
    ----------
    relation : str
        A string representing the desired comparison operator.
        Expected values are '>', '<', '>=', '<=', '==', or '!='.
    """
    relation = relation.strip()
    ops = {
        ">": np.greater,
        "<": np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
        "==": np.equal,
        "!=": np.not_equal,
    }
    if relation in ops:
        return ops[relation]
    raise ValueError(f"Unknown relation {relation}")
