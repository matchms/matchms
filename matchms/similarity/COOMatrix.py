from typing import Optional
import numpy as np


class COOMatrix:
    """Container for a COOMatrix.
    This container is needed next to the scipy coo_matrix, since the scipy coo_matrix does not support datatypes
    with multiple entries like the cosine scores and matches."""
    def __init__(self, row_idx, column_idx, scores, scores_dtype: Optional[np.dtype] = None):
        if not len(row_idx) == len(column_idx) == len(scores):
            raise ValueError("row_idx, column_idx, and scores must have the same length")

        if isinstance(row_idx, list):
            self.row= np.array(row_idx, dtype=np.int_)
        elif isinstance(row_idx, np.ndarray):
            self.row = row_idx
        else:
            raise TypeError("row_idx must be either a list or np.ndarray")

        if isinstance(row_idx, list):
            self.column = np.array(column_idx, dtype=np.int_)
        elif isinstance(column_idx, np.ndarray):
            self.column = column_idx
        else:
            raise TypeError("column_idx must be either a list or np.ndarray")

        if isinstance(scores, list):
            if scores_dtype is None:
                raise ValueError("scores_dtype must be provided if not yet a numpy array")
            self.scores = np.array(scores,
                                   dtype=scores_dtype)
        elif isinstance(scores, np.ndarray):
            self.scores = scores

    def to_dense_matrix(self, nr_of_rows, nr_of_columns) -> np.ndarray:
        array = np.zeros((nr_of_rows, nr_of_columns),
                         dtype=self.scores.dtype)
        if len(self.scores) > 0:
            array[self.row, self.column] = self.scores.reshape(-1)
        return array

    def __str__(self):
        return f"COOMatrix({self.row}, {self.column}, {self.scores})"
