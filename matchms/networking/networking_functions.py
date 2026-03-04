"""Helper functions to build and handle spectral networks"""

import numpy as np
from scipy.sparse import coo_array, csr_array


def get_top_hits_matrix(matrix: np.ndarray, top_n: int, ignore_diagonal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    top_n = min(top_n, matrix.shape[1] - (1 if ignore_diagonal else 0))
    if ignore_diagonal:
        matrix = matrix.copy()
        np.fill_diagonal(matrix, -np.inf)
    indexes_of_top_scores = np.argpartition(matrix, -top_n, axis=1)[:, -top_n:]
    top_scores = np.take_along_axis(matrix, indexes_of_top_scores, axis=1)

    # Sort descending within each row
    sort_order = np.argsort(top_scores, axis=1)[:, ::-1]
    indexes_of_top_scores = np.take_along_axis(indexes_of_top_scores, sort_order, axis=1)
    highest_scores = np.take_along_axis(top_scores, sort_order, axis=1)
    return highest_scores, indexes_of_top_scores


def get_top_hits_coo_array(
    coo_array: coo_array, top_n: int, ignore_diagonal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if coo_array.shape is None:
        raise ValueError("Shape of input COO array must be defined.")
    top_n = min(top_n, coo_array.shape[1] - (1 if ignore_diagonal else 0))

    csr = csr_array(coo_array)
    n_rows = coo_array.shape[0]

    highest_scores = np.full((n_rows, top_n), np.nan)
    indexes_of_top_scores = np.full((n_rows, top_n), -1, dtype=int)

    for r in range(n_rows):
        start, end = csr.indptr[r], csr.indptr[r + 1]
        row_data = csr.data[start:end].copy()
        row_cols = csr.indices[start:end]

        if ignore_diagonal:
            diag_mask = row_cols == r
            row_data[diag_mask] = -np.inf

        row_top_n = min(top_n, len(row_data))
        if row_top_n == len(row_data):
            top_idx = np.argsort(row_data)[::-1]  # sort all
        else:
            top_idx = np.argpartition(row_data, -row_top_n)[-row_top_n:]
            top_idx = top_idx[np.argsort(row_data[top_idx])[::-1]]  # sort top_n

        highest_scores[r, :row_top_n] = row_data[top_idx]
        indexes_of_top_scores[r, :row_top_n] = row_cols[top_idx]

    return highest_scores, indexes_of_top_scores
