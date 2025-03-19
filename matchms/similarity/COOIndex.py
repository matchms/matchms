from typing import Iterator, Tuple, Union
import numpy as np


class COOIndex:
    """Row and column indexes for a COO matrix

    Can be used as a mask to specify between which pairs similarity scores should be computed"""
    def __init__(self, idx_row: np.ndarray, idx_col: np.ndarray):
        self._idx_row = idx_row
        self._idx_col = idx_col
        if not (self._idx_row.ndim == 1 and self._idx_col.ndim == 1):
            raise ValueError('idx_row and idx_col must be 1D')
        if self._idx_row.shape[0] != self._idx_col.shape[0]:
            raise ValueError('idx_row and idx_col must have same length')
        self.length = self._idx_row.shape[0]

    @property
    def idx_row(self) -> np.ndarray:
        return self._idx_row

    @property
    def idx_col(self) -> np.ndarray:
        return self._idx_col

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return zip(self._idx_row, self._idx_col)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: Union[int, slice]) -> Union[Tuple[np.ndarray, np.ndarray], "COOIndex"]:
        if isinstance(item, int):
            return self._idx_row[item], self._idx_col[item]
        elif isinstance(item, slice):
            return COOIndex(self._idx_row[item], self._idx_col[item])  # Return a COOIndex subset
        else:
            raise TypeError("Index must be an integer or a slice")
