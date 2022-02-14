# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.sputils import get_index_dtype


_slicing_not_implemented_msg = "Wrong slicing, or option not yet implemented"


class StackedSparseScores:
    """ 2.5D sparse matrix in COO-style with multiple possible entries per i-j-position.

    Add description...

    Parameters
    ----------
    n_row
        Number of rows of sparse array.
    n_cols
        Number of colums of sparse array.

    Code example:

    .. code-block:: python
        import numpy as np
        from matchms import StackedSparseScores

        scores1 = np.random.random((12, 10))
        scores2 = np.random.random((12, 10))

        scores_array = StackedSparseScores(12, 10)
        scores_array.add_dense_matrix(scores1, "scores_1")
        scores_array.filter_by_range("scores_1", low=0.5)

        # Add second scores and filter
        scores_array.add_dense_matrix(scores2, "scores_2")
        scores_array.filter_by_range("scores_2", low=0.1, high=0.4)

        scores2_after_filtering = scores_array.toarray("scores_2")

    """
    def __init__(self, n_row, n_col, name=None):
        self.__n_row = n_row
        self.__n_col = n_col
        idx_dtype = get_index_dtype(maxval=max(n_row, n_col))
        self._row = np.array([], dtype=idx_dtype)
        self._col = np.array([], dtype=idx_dtype)
        self._data = OrderedDict()
        if name is not None:
            self._data[name] = np.empty(0)

    def guess_score_name(self):
        if len(self._data.keys()) == 1:
            return list(self._data.keys())[0]
        if len(self._data.keys()) == 0:
            raise ValueError("Array is empty.")
        raise KeyError("Name of score is required.")

    def __repr__(self):
        msg = f"<{self.shape[0]}x{self.shape[1]}x{self.shape[2]} stacked sparse array" \
            f" containing scores for {self.score_names}" \
            f" with {len(self._data)} stored elements in COOrdinate format>"
        return msg

    def __str__(self):
        msg = f"StackedSparseScores array of shape {self.shape}" \
            f" containing scores for {self.score_names}."
        return msg

    def __setitem__(self, key, d):
        # Typical COO method (e.g. below) would not be safe for stacked array.
        raise NotImplementedError
        # row, col, name = self._validate_indices(key)
        # self.row = np.append(self.row, row)
        # self.col = np.append(self.col, col)
        # self._data[name] = np.append(self._data[name], d)

    def __getitem__(self, key):
        row, col, name = self._validate_indices(key)
        r, c, d = self._getitem_method(row, col, name)
        if len(r) == 0:
            return np.array([0])
        if r.shape[0] == 1:
            return d
        return r, c, d

    def _getitem_method(self, row, col, name):
        # e.g.: matrix[3, 7, "score_1"]
        if isinstance(row, int) and isinstance(col, int):
            idx = np.where((self.row == row) & (self.col == col))
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[3, :, "score_1"]
        if isinstance(row, int) and isinstance(col, slice):
            if not col.start == col.stop == col.step is None:
                raise IndexError(_slicing_not_implemented_msg)
            idx = np.where(self.row == row)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[:, 7, "score_1"]
        if isinstance(row, slice) and isinstance(col, int):
            if not row.start == row.stop == row.step is None:
                raise IndexError(_slicing_not_implemented_msg)
            idx = np.where(self.col == col)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # matrix[:, :, "score_1"]
        if isinstance(row, slice) and isinstance(col, slice):
            if not row.start == row.stop == row.step is None:
                raise IndexError(_slicing_not_implemented_msg)
            if not col.start == col.stop == col.step is None:
                raise IndexError(_slicing_not_implemented_msg)
            return self.row, self.col, self._slicing_data(name)
        if row == col is None and isinstance(name, str):
            return self.row, self.col, self._slicing_data(name)
        raise IndexError(_slicing_not_implemented_msg)

    def _slicing_data(self, name, idx=None):
        if isinstance(name, slice) and len(self.score_names) == 1:
            name = self.score_names[0]
        if isinstance(name, str):
            if idx is None:
                return self.data[name]
            return self.data[name][idx]
        if isinstance(name, slice) and name.start == name.stop == name.step is None:
            if idx is None:
                return [value.copy() for _, value in self._data.items()]
            return [value[idx].copy() for _, value in self._data.items()]
        raise IndexError(_slicing_not_implemented_msg)

    def _validate_indices(self, key):
        m, n, _ = self.shape
        row, col, name = _unpack_index(key)
        if row == col is None and isinstance(name, str):
            return row, col, name

        if isinstance(name, int):
            name = self.score_names[name]

        if isinstance(row, int):
            if row < -m or row >= m:
                raise IndexError(f"row index ({row}) out of range")
            if row < 0:
                row += m
        elif not isinstance(row, slice):
            row = self._asindices(row, m)

        if isinstance(col, int):
            if col < -n or col >= n:
                raise IndexError(f"column index ({col}) out of range")
            if col < 0:
                col += n
        elif not isinstance(col, slice):
            col = self._asindices(col, n)

        return row, col, name

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.
        Subclasses that need special validation can override this method.
        """
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be 1 or 2')

        if x.size == 0:
            return x

        # Check bounds
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError(f"Index ({max_indx}) out of range")

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError(f"Index ({min_indx}) out of range")
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    @property
    def row(self):
        return self._row.copy()

    @property
    def col(self):
        return self._col.copy()

    @property
    def data(self):
        return self._data.copy()

    @property
    def shape(self):
        return tuple((self.__n_row, self.__n_col, len(self._data)))

    @property
    def score_names(self):
        return list(self._data.keys())

    def add_dense_matrix(self, matrix: np.ndarray,
                         name: str):
        """Add dense array (numpy array) to stacked sparse scores.

        If the StackedSparseScores is still empty, the full dense matrix will
        be added, unless threshold are set by `low` and `high` values.
        If the StackedSparseScores already contains one or more scores, than only
        those values of the input matrix will be added which have the same position
        as already existing entries!

        Parameters
        ----------
        matrix
            Input (dense) array, such as numpy array to be added to the stacked sparse
            scores.
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.

        """
        if len(matrix.dtype) > 1:  # if structured array
            for dtype_name in matrix.dtype.names:
                self._add_dense_matrix(matrix[dtype_name], name + "_" + dtype_name)
        else:
            self._add_dense_matrix(matrix, name)

    def _add_dense_matrix(self, matrix, name):
        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self._data.keys()):
            # Add first (sparse) array of scores
            (idx_row, idx_col) = np.where(matrix)
            self._row = idx_row
            self._col = idx_col
            self._data = {name: matrix[idx_row, idx_col]}
        else:
            # Add new stack of scores
            self._data[name] = matrix[self.row, self.col]

    def add_coo_matrix(self, coo_matrix, name):
        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self._data.keys()):
            # Add first (sparse) array of scores
            self._data = {name: coo_matrix.data}
            self._row = coo_matrix.row
            self._col = coo_matrix.col
            self.__n_row, self.__n_col = coo_matrix.shape
        else:
            # TODO move into logger warning rather than assert
            assert len(np.setdiff1d(coo_matrix.row, self.row)) == 0, "New, unknown row indices"
            assert len(np.setdiff1d(coo_matrix.col, self.col)) == 0, "New, unknown col indices"
            new_entries = []
            # TODO: numbafy...
            for i, coo_row_id in enumerate(coo_matrix.row):
                idx = np.where((self.row == coo_row_id)
                               & (self.col == coo_matrix.col[i]))[0][0]
                new_entries.append(idx)

            self._data[name] = np.zeros((len(self.row)), dtype=coo_matrix.dtype)
            self._data[name][new_entries] = coo_matrix.data

    def filter_by_range(self, name: str = None,
                        low=-np.inf, high=np.inf,
                        above_operator='>',
                        below_operator='<'):
        """Remove all scores for which the score `name` is outside the given range.

        Add description

        Parameters
        ----------

        """
        # pylint: disable=too-many-arguments
        if name is None:
            name = self.guess_score_name()
        above_operator = _get_operator(above_operator)
        below_operator = _get_operator(below_operator)
        idx = np.where(above_operator(self._data[name], low)
                       & below_operator(self._data[name], high))
        self._col = self.col[idx]
        self._row = self.row[idx]
        for key, value in self._data.items():
            self._data[key] = value[idx]

    def to_array(self, name):
        array = np.zeros((self.__n_row, self.__n_col),
                         dtype=self._data[name].dtype)
        array[self.row, self.col] = self._data[name]
        return array

    def to_coo(self, name):
        return coo_matrix((self._data[name], (self.row, self.col)),
                          shape=(self.__n_row, self.__n_col))

    def get_indices(self, name=None, threshold=-np.Inf):
        # TODO: refactor or remove this method
        if name is None:
            name = self.guess_score_name()
        idx = np.where(self._data[name] > threshold)
        return self.row[idx], self.col[idx]


def _unpack_index(index):
    if isinstance(index, tuple):
        if len(index) == 3:
            row, col, name = index
            if name is None:
                name = slice(None)
        elif len(index) == 2:
            row, col, name = index[0], index[1], slice(None)
        elif len(index) == 1:
            row, col, name = index[0], slice(None), slice(None)
        else:
            raise IndexError("Invalid number of indices")
    elif isinstance(index, str):
        row, col, name = None, None, index
    return row, col, name


def _get_operator(relation: str):
    relation = relation.strip()
    ops = {'>': np.greater,
           '<': np.less,
           '>=': np.greater_equal,
           '<=': np.less_equal}
    if relation in ops:
        return ops[relation]
    raise ValueError(f"Unknown relation {relation}")
