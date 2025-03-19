from typing import Union, Tuple, Iterator

import numpy as np


class Mask:
    """Row and column indexes for a matrix

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

    def __getitem__(self, item: Union[int, slice]) -> Union[Tuple[np.ndarray, np.ndarray], "Mask"]:
        if isinstance(item, int):
            return self._idx_row[item], self._idx_col[item]
        elif isinstance(item, slice):
            return Mask(self._idx_row[item], self._idx_col[item])  # Return a Mask subset
        else:
            raise TypeError("Index must be an integer or a slice")
