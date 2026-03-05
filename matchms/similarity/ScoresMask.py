import logging
from typing import Iterator, Tuple, Union
import numpy as np
from scipy.sparse import coo_array


logger = logging.getLogger("matchms")


class ScoresMask:
    """Row and column indexes for a COO matrix

    Can be used as a mask to specify between which pairs similarity scores should be computed"""

    def __init__(self, idx_row: np.ndarray, idx_col: np.ndarray, nrows: int, ncols: int):
        self._idx_row = idx_row
        self._idx_col = idx_col
        if not (self._idx_row.ndim == 1 and self._idx_col.ndim == 1):
            raise ValueError("idx_row and idx_col must be 1D")
        if self._idx_row.shape[0] != self._idx_col.shape[0]:
            raise ValueError("idx_row and idx_col must have same length")
        self.length = self._idx_row.shape[0]
        if idx_row.size > 0 and idx_row.max() >= nrows:
            raise ValueError(f"idx_row contains index {idx_row.max()} >= nrows ({nrows})")
        if idx_col.size > 0 and idx_col.max() >= ncols:
            raise ValueError(f"idx_col contains index {idx_col.max()} >= ncols ({ncols})")
        self.nrows = nrows
        self.ncols = ncols

    @classmethod
    def from_matrix(cls, scores_matrix: np.ndarray, operation: str, value) -> "ScoresMask":
        if not isinstance(scores_matrix, np.ndarray) or scores_matrix.ndim != 2:
            raise ValueError("Input must be a dense matrix. Use ScoresMask.from_coo_array() for COO arrays.")
        operator = _get_operator(operation)
        mask = operator(scores_matrix, value)
        rows, cols = np.where(mask)
        return cls(rows, cols, nrows=scores_matrix.shape[0], ncols=scores_matrix.shape[1])

    @classmethod
    def from_coo_array(cls, scores_coo_array: coo_array, operation: str, value) -> "ScoresMask":
        if not isinstance(scores_coo_array, coo_array):
            raise ValueError("Input must be a COO array. Use ScoresMask.from_matrix() for dense matrices.")
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
        if scores_coo_array.shape is None:
            raise ValueError("Shape of COO array must be defined to create ScoresMask.")
        return cls(rows, cols, nrows=scores_coo_array.shape[0], ncols=scores_coo_array.shape[1])

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
            return ScoresMask(self._idx_row[item], self._idx_col[item], nrows=self.nrows, ncols=self.ncols)
        raise TypeError("Index must be an integer or a slice")

    def _encode(self) -> np.ndarray:
        return self._idx_row.astype(np.int64) * self.ncols + self._idx_col

    def _decode(self, encoded: np.ndarray):
        return encoded // self.ncols, encoded % self.ncols

    def __or__(self, other: "ScoresMask") -> "ScoresMask":
        combined = np.union1d(self._encode(), other._encode())
        rows, cols = self._decode(combined)
        return ScoresMask(rows, cols, self.nrows, self.ncols)

    def __and__(self, other: "ScoresMask") -> "ScoresMask":
        common = np.intersect1d(self._encode(), other._encode())
        rows, cols = self._decode(common)
        return ScoresMask(rows, cols, self.nrows, self.ncols)

    def __sub__(self, other: "ScoresMask") -> "ScoresMask":
        diff = np.setdiff1d(self._encode(), other._encode())
        rows, cols = self._decode(diff)
        return ScoresMask(rows, cols, self.nrows, self.ncols)

    @property
    def coverage(self) -> float:
        return self.length / (self.nrows * self.ncols)


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
