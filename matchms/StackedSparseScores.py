# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib import recfunctions
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
        scores_array = scores_array.filter_by_range("scores_1", low=0.5)

        # Add second scores and filter
        scores_array.add_dense_matrix(scores2, "scores_2")
        scores_array = scores_array.filter_by_range("scores_2", low=0.1, high=0.4)

        # Scores can be accessed using (limited) slicing capabilities
        scores_array[3, 4]  # => scores_1 and scores_2 at position row=3, col=4
        scores_array[3, :]  # => tuple with row, col, scores for all entries in row=3
        scores_array[:, 2]  # => tuple with row, col, scores for all entries in col=2
        scores_array[3, :, 0]  # => tuple with row, col, scores_1 for all entries in row=3
        scores_array[3, :, "scores_1"]  # => same as the one before

        # Scores can also be converted to a dense numpy array:
        scores2_after_filtering = scores_array.to_array("scores_2")

    """
    def __init__(self, n_row, n_col):
        self.__n_row = n_row
        self.__n_col = n_col
        idx_dtype = get_index_dtype(maxval=max(n_row, n_col))
        self.row = np.array([], dtype=idx_dtype)
        self.col = np.array([], dtype=idx_dtype)
        self.data = None

    def __repr__(self):
        msg = f"<{self.shape[0]}x{self.shape[1]}x{self.shape[2]} stacked sparse array" \
            f" containing scores for {self.score_names}" \
            f" with {len(self.row)} stored elements in COOrdinate format>"
        return msg

    def __str__(self):
        msg = f"StackedSparseScores array of shape {self.shape}" \
            f" containing scores for {self.score_names}."
        return msg

    def __eq__(self, other):
        if self.score_names != other.score_names:
            return False
        if self.shape != other.shape:
            return False
        if np.any(self.row != other.row):
            return False
        if np.any(self.col != other.col):
            return False
        if np.any(self.data != other.data):
            return False
        return True

    def __setitem__(self, key, d):
        # Typical COO method (e.g. below) would not be safe for stacked array.
        raise NotImplementedError

    def __getitem__(self, key):
        row, col, name = self._validate_indices(key)
        r, c, d = self._getitem_method(row, col, name)
        if isinstance(row, int) and isinstance(col, int):
            if len(r) == 0:
                return np.array([0])
            return d
        return r, c, d

    def _getitem_method(self, row, col, name):
        # e.g.: matrix[3, 7, "score_1"]
        if isinstance(row, int) and isinstance(col, int):
            idx = np.where((self.row == row) & (self.col == col))
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[3, :, "score_1"]
        if isinstance(row, int) and isinstance(col, slice):
            self._is_implemented_slice(col)
            idx = np.where(self.row == row)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # e.g.: matrix[:, 7, "score_1"]
        if isinstance(row, slice) and isinstance(col, int):
            self._is_implemented_slice(row)
            idx = np.where(self.col == col)
            return self.row[idx], self.col[idx], self._slicing_data(name, idx)
        # matrix[:, :, "score_1"]
        if isinstance(row, slice) and isinstance(col, slice):
            self._is_implemented_slice(row)
            self._is_implemented_slice(col)
            return self.row, self.col, self._slicing_data(name)
        if row == col is None and isinstance(name, str):
            return self.row, self.col, self._slicing_data(name)
        raise IndexError(_slicing_not_implemented_msg)

    def _is_implemented_slice(self, input_slice):
        # Currently slices like matrix[2:4, :] or not implemented
        if not input_slice.start == input_slice.stop == input_slice.step is None:
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
                return self.data
            return self.data[idx]
        raise IndexError(_slicing_not_implemented_msg)

    def _validate_indices(self, key):
        def validate_index(index, shape):
            if isinstance(index, int):
                if index < -shape or index >= shape:
                    raise IndexError(f"Index ({index}) out of range")
                if index < 0:
                    index += shape
            elif not isinstance(index, slice):
                index = self._asindices(index, shape)
            return index

        m, n, _ = self.shape
        row, col, name = _unpack_index(key)
        if row == col is None and isinstance(name, str):
            return row, col, name

        if isinstance(name, int):
            name = self.score_names[name]

        row = validate_index(row, m)
        col = validate_index(col, n)
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
    def shape(self):
        return self.__n_row, self.__n_col, len(self.score_names)

    @property
    def score_names(self):
        if self.data is None:
            return []
        if self.data.dtype.names is None:
            return [self.data.dtype.str]
        return self.data.dtype.names

    def clone(self):
        """ Returns clone (deepcopy) of StackedSparseScores instance."""
        cloned_array = StackedSparseScores(self.__n_row, self.__n_col)
        cloned_array.row = self.row
        cloned_array.col = self.col
        cloned_array.data = self.data
        return cloned_array

    def add_dense_matrix(self, matrix: np.ndarray,
                         name: str):
        """Add dense array (numpy array) to stacked sparse scores.

        If the StackedSparseScores is still empty, the full dense matrix will
        be added.
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
        if matrix is None:
            self.data = np.array([])
        elif len(matrix.dtype) > 1:  # if structured array
            for dtype_name in matrix.dtype.names:
                self._add_dense_matrix(matrix[dtype_name], name + "_" + dtype_name)
        else:
            self._add_dense_matrix(matrix, name)

    def _add_dense_matrix(self, matrix, name):
        if matrix.dtype.type == np.void:
            input_dtype = matrix.dtype[0]
        else:
            input_dtype = matrix.dtype
        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self.score_names):
            # Add first (sparse) array of scores
            (idx_row, idx_col) = np.where(matrix)
            self.row = idx_row
            self.col = idx_col
            self.data = np.array(matrix[idx_row, idx_col], dtype=[(name, input_dtype)])
        else:
            # Add new stack of scores
            self.data = recfunctions.append_fields(self.data, name, matrix[self.row, self.col],
                                                    dtypes=input_dtype, fill_value=0).data

    def guess_score_name(self):
        if len(self.score_names) == 1:
            return self.score_names[0]
        if len(self.score_names) == 0:
            raise KeyError("Array is empty.")
        raise KeyError("Name of score is required.")

    def add_coo_matrix(self, coo_matrix, name):
        """Add sparse matrix (scipy COO-matrix) to stacked sparse scores.

        If the StackedSparseScores is still empty, the full sparse matrix will
        be added.
        If the StackedSparseScores already contains one or more scores, than only
        those values of the input matrix will be added which have the same position
        as already existing entries!

        Parameters
        ----------
        matrix
            Input sparse matrix (COO-style containing .row, .col, .data) to be
            added to the stacked sparse scores.
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.

        """
        if self.shape[2] == 0 or (self.shape[2] == 1 and name in self.score_names):
            # Add first (sparse) array of scores
            self.data = np.array(coo_matrix.data, dtype=[(name, coo_matrix.dtype)])
            self.row = coo_matrix.row.copy()
            self.col = coo_matrix.col.copy()
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

            self.data = recfunctions.append_fields(self.data, name,
                                                    np.zeros((len(self.row)), dtype=coo_matrix.dtype),
                                                    fill_value=0).data
            self.data[name][new_entries] = coo_matrix.data

    def add_sparse_data(self, data: np.ndarray, name: str,
                        row=None, col=None):
        """Add sparse data to stacked sparse scores.

        The given data must be of the same dimension as the current row/col values.
        Or (if no data is present yet) row and col indices must be provided as well.

        Parameters
        ----------
        data
            Input data (1D array).
        name
            Name of the score which is added. Will later be used to access and address
            the added scores, for instance via `sss_array.toarray("my_score_name")`.
        """
        if data is None:
            self.data = np.array([])
        elif len(data.dtype) > 1:  # if structured array
            for dtype_name in data.dtype.names:
                if self.data is None:
                    self._add_sparse_data(data[dtype_name], row, col, name + "_" + dtype_name)
                else:
                    self._add_sparse_data_to_existing(data[dtype_name], name + "_" + dtype_name)
        elif self.data is None:
            self._add_sparse_data(data, row, col, name)
        else:
            self._add_sparse_data_to_existing(data, name)

    def _add_sparse_data(self, data, row, col, name):
        assert name not in self.score_names, "Scores of 'name' are already found in array"
        self.row = row
        self.col = col
        self.data = np.array(data, dtype=[(name, data.dtype)])

    def _add_sparse_data_to_existing(self, data, name):
        assert data.shape[0] == self.row.shape[0], \
            "Data must be of same size as number of sparse values in the array"
        assert name not in self.score_names, "Scores of 'name' are already found in array"
        self.data = recfunctions.append_fields(self.data, name, data,
                                                dtypes=data.dtype, fill_value=0).data

    def filter_by_range(self, name: str = None,
                        low=-np.inf, high=np.inf,
                        above_operator='>',
                        below_operator='<'):
        """Remove all scores for which the score `name` is outside the given range.

        Parameters
        ----------
        name
            Name of the score which is used for filtering. Run `.score_names` to
            see all scores scored in the sparse array.
        low
            Lower threshold below which all scores will be removed.
        high
            Upper threshold above of which all scores will be removed.
        above_operator
            Define operator to be used to compare against `low`. Default is '>'.
            Possible choices are '>', '<', '>=', '<='.
        below_operator
            Define operator to be used to compare against `high`. Default is '<'.
            Possible choices are '>', '<', '>=', '<='.
        """
        # pylint: disable=too-many-arguments
        above_operator = _get_operator(above_operator)
        below_operator = _get_operator(below_operator)
        if name is None:
            name = self.guess_score_name()
        idx = np.where(above_operator(self.data[name], low)
                       & below_operator(self.data[name], high))
        cloned_array = StackedSparseScores(self.__n_row, self.__n_col)
        cloned_array.col = self.col[idx]
        cloned_array.row = self.row[idx]
        cloned_array.data = self.data[idx]
        return cloned_array

    def to_array(self, name=None):
        """Return scores as (non-sparse) numpy array.

        Parameters
        ----------
        name
            Name of the score that should be returned (if multiple scores are stored).
            If set to None (default) a 3D array with all scores will be returned.
        """
        if self.data is None:
            return None
        if name is None and self.shape[2] == 1:
            name = self.score_names[0]
        if isinstance(name, str):
            array = np.zeros((self.__n_row, self.__n_col),
                             dtype=self.data[name].dtype)
            array[self.row, self.col] = self.data[name]
            return array
        array = np.zeros((self.__n_row, self.__n_col),
                         dtype=self.data.dtype)
        array[self.row, self.col] = self.data
        return array

    def to_coo(self, name):
        return coo_matrix((self.data[name], (self.row, self.col)),
                          shape=(self.__n_row, self.__n_col))


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
    elif isinstance(index, int):
        row, col, name = index, slice(None), slice(None)
    else:
        raise IndexError(_slicing_not_implemented_msg)
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
