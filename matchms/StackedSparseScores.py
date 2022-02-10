# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.sputils import get_index_dtype


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

        matrix = StackedSparseScores(12, 10)
        matrix.add_dense_matrix(scores1, "scores_1")
        matrix.filter_by_range("scores_1", low=0.5)

        # Add second scores and filter
        matrix.add_dense_matrix(scores2, "scores_2")
        matrix.filter_by_range("scores_2", low=0.1, high=0.4)

        scores2_after_filtering = matrix.toarray("scores_2")

    """
    def __init__(self, n_row, n_col):
        self.__n_row = n_row
        self.__n_col = n_col
        idx_dtype = get_index_dtype(maxval=max(n_row, n_col))
        self.row = np.array([], dtype=idx_dtype)
        self.col = np.array([], dtype=idx_dtype)
        self._data = {}
        self.metadata = {0: {"name": None}}

    def _guess_name(self):
        if len(self._data.keys()) == 1:
            return list(self._data.keys())[0]
        raise KeyError("Name of score is required.")

    def __setitem__(self, dim_name, d):
        r, c, name = dim_name
        assert name in self._data, "Name must exist..."
        # assert len(r) == len(c) == len(d), "Elements must be of same length."
        self.row = np.append(self.row, r)
        self.col = np.append(self.col, c)
        self._data[name] = np.append(self._data[name], d)

    def __getitem__(self, key):
        print(key)
        print(self._validate_indices(key))
        row, col, name = self._validate_indices(key)

        if isinstance(row, int):
            idx_row = np.where(self.row == row)
        elif isinstance(row, slice):
            if row.start == row.stop == row.step is None:
                idx_row = np.arange(0, len(self.row))
            else:
                raise IndexError("This slicing option is not yet implemented")
        if isinstance(col, int):
            idx_col = np.where(self.col == col)
        elif isinstance(col, slice):
            if col.start == col.stop == col.step is None:
                idx_col = np.arange(0, len(self.col))
            else:
                raise IndexError("This slicing option is not yet implemented")
        print(idx_row, idx_col)
        idx = np.intersect1d(idx_row, idx_col)
        print(idx)
        if idx.shape[0] > 1:
            return self.row[idx], self.col[idx], self._data[name][idx]
        if idx.shape[0] == 0:
            return 0
        return self._data[name][idx][0]

    def _validate_indices(self, key):
        m, n, _ = self.shape
        row, col, name = _unpack_index(key)

        if name is None:
            name = self._guess_name()

        if isinstance(row, int):
            if row < -m or row >= m:
                raise IndexError('row index (%d) out of range' % row)
            if row < 0:
                row += m
        elif not isinstance(row, slice):
            row = self._asindices(row, m)

        if isinstance(col, int):
            if col < -n or col >= n:
                raise IndexError('column index (%d) out of range' % col)
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
            raise IndexError('index (%d) out of range' % max_indx)

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError('index (%d) out of range' % min_indx)
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    @property
    def data(self):
        return self._data.copy()

    @property
    def shape(self):
        return tuple((self.__n_row, self.__n_col, len(self._data)))

    def eliminate_zeros(self):
        """Remove zero entries from the matrix
        This is an *in place* operation
        """
        mask = self._data != 0
        self._data = self._data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    def eliminate_below_threshold(self, threshold):
        """Remove entries below threshold from the matrix."""
        mask = self._data < threshold
        self._data = self._data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    def add_dense_matrix(self, matrix: np.ndarray,
                         name: str,
                         low: float = None, high: float = None):
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
        low
            Set to numerical value if a lower threshold should be applied. The default is None.
        high
            Set to numerical value if an upper threshold should be applied. The default is None.

        """
        if self.shape[2] == 0:
            # Add first (sparse) array of scores
            if low is None:
                low = -np.inf
            if high is None:
                high = np.inf
            (idx_row, idx_col) = np.where((matrix > low) & (matrix < high))
            self.row = idx_row
            self.col = idx_col
            self._data = {name: matrix[idx_row, idx_col]}
        elif low is None and high is None:
            # Add new stack of scores
            self._data[name] = matrix[self.row, self.col]
        else:
            if low is None:
                low = -np.inf
            elif high is None:
                high = np.inf
            self._data[name] = matrix[self.row, self.col]
            self.filter_by_range(name, low=low, high=high)

    def add_coo_matrix(self, coo_matrix, name):
        if self.shape[2] == 0:
            # Add first (sparse) array of scores
            self._data = {name: coo_matrix.data}
            self.row = coo_matrix.row
            self.col = coo_matrix.col
            self.__n_row, self.__n_col = coo_matrix.shap_data
        else:
            # TODO move into logger warning rather than assert
            assert len(np.setdiff1d(coo_matrix.row, self.row)) == 0, "New, unknown row indices"
            assert len(np.setdiff1d(coo_matrix.col, self.col)) == 0, "New, unknown col indices"
            new_entries = []
            # TODO: numbafy...
            for i, coo_row_id in enumerate(coo_matrix.row):
                idx = np.where((self.row == coo_row_id) \
                               & (self.col == coo_matrix.col[i]))[0][0]
                new_entries.append(idx)

        self._data[name] = np.zeros((len(self.row)))
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
        if name is None:
            name = self._guess_name()
        above_operator = _get_operator(above_operator)
        below_operator = _get_operator(below_operator)
        idx = np.where(above_operator(self._data[name], low) &\
                       below_operator(self._data[name], high))
        self.col = self.col[idx]
        self.row = self.row[idx]
        for key, value in self._data.items():
            self._data[key] = value[idx]

    def toarray(self, name):
        return coo_matrix((self._data[name], (self.row, self.col)),
                          shape=(self.__n_row, self.__n_col)).toarray()

    def get_indices(self, name=None, threshold=-np.Inf):
        if name is None:
            name = self._guess_name()
        idx = np.where(self._data[name] > threshold)
        return self.row[idx], self.col[idx]


def _unpack_index(index):
    if isinstance(index, tuple):
        if len(index) == 3:
            row, col, name = index
        elif len(index) == 2:
            row, col, name = index[0], index[1], None
        elif len(index) == 1:
            row, col, name = index[0], slice(None), None
        else:
            raise IndexError('invalid number of indices')
    return row, col, name


def _get_operator(relation: str):
    relation = relation.strip()
    ops = {'>': np.greater,
           '<': np.less,
           '>=': np.greater_equal,
           '<=': np.less_equal}
    if relation in ops.keys():
        return ops[relation]
    raise ValueError("Unknown relation %s", relation)
