import numpy as np
import pytest
from scipy.sparse import coo_matrix
from matchms.StackedSparseScores import StackedSparseScores


@pytest.fixture
def sparse_array():
    arr = np.arange(0, 120).reshape(12, 10)
    arr[arr % 2 == 1] = 0
    arr[arr % 4 == 0] = 0
    return arr


def test_sss_matrix_empty():
    matrix = StackedSparseScores(200, 100)
    assert matrix.shape == (200, 100, 0)
    assert matrix.score_names == []


def test_sss_matrix_add_dense():
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    assert matrix.shape == (12, 10, 0)
    matrix.add_dense_matrix(arr, "test_score")
    assert matrix.shape == (12, 10, 1)
    assert np.all(matrix.data["test_score"] == np.arange(1, 120))


def test_sss_matrix_add_empty_array():
    arr = np.array([[False, False], [False, False]])
    matrix = StackedSparseScores(2, 2)
    assert matrix.shape == (2, 2, 0)
    matrix.add_dense_matrix(arr, "test_score")
    assert matrix.shape == (2, 2, 1)
    assert len(matrix.data) == 0
    assert np.all(matrix.to_array().tolist() == arr)


def test_sss_matrix_no_setter():
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(arr, "test_score")
    with pytest.raises(NotImplementedError):
        matrix[1, 2] = 5


def test_sss_matrix_class_name(sparse_array):
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(sparse_array, "test_score")
    msg = "<12x10x1 stacked sparse array containing scores for ('test_score',)" \
        " with 30 stored elements in COOrdinate format>"
    assert matrix.__repr__() == msg
    msg2 = "StackedSparseScores array of shape (12, 10, 1) containing scores for" \
        " ('test_score',)."
    assert str(matrix) == msg2


def test_sss_matrix_add_coo(sparse_array):
    sparse_array = coo_matrix(sparse_array)
    matrix = StackedSparseScores(12, 10)
    assert matrix.shape == (12, 10, 0)
    matrix.add_coo_matrix(sparse_array, "test_score")
    assert matrix.shape == (12, 10, 1)
    assert np.all(matrix.data["test_score"] == np.arange(2, 120, 4))
    r, c, v = matrix["test_score"]
    assert np.all(v == np.arange(2, 120, 4))
    expected = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3,
                         4, 4, 5, 5, 5, 6, 6, 7, 7, 7,
                         8, 8, 9, 9, 9, 10, 10, 11, 11, 11])
    assert np.all(r == expected)
    assert np.all(c[:6] == np.array([2, 6, 0, 4, 8, 2]))


def test_sss_matrix_add_coo_2_times(sparse_array):
    sparse_array1 = coo_matrix(sparse_array.astype(np.int32))
    sparse_array[sparse_array % 10 == 0] = 0
    sparse_array = sparse_array/2
    sparse_array2 = coo_matrix(sparse_array.astype(np.int32))

    matrix = StackedSparseScores(12, 10)
    matrix.add_coo_matrix(sparse_array1, "scores1")
    matrix.add_coo_matrix(sparse_array2, "scores2")
    assert matrix.shape == (12, 10, 2)
    assert np.all(matrix.data["scores1"] == np.arange(2, 120, 4))
    expected = np.array([1, 3, 0, 7, 9, 11, 13, 0, 17, 19, 21,
                         23, 0, 27, 29, 31, 33, 0, 37, 39, 41,
                         43, 0, 47, 49, 51, 53, 0, 57, 59])
    assert np.all(matrix.data["scores2"] == expected)
    assert matrix.data["scores1"].dtype == np.int32
    assert matrix.data["scores2"].dtype == np.int32

    # Run filter
    matrix = matrix.filter_by_range("scores2", low=0, high=40)
    expected = np.array([2, 6, 14, 18, 22, 26, 34, 38, 42,
                         46, 54, 58, 62, 66, 74, 78])
    assert np.all(matrix.data["scores1"] == expected)
    expected = np.array([1, 3, 7, 9, 11, 13, 17, 19, 21,
                         23, 27, 29, 31, 33, 37, 39])
    assert np.all(matrix.data["scores2"] == expected)

    msg = "Name of score is required."
    with pytest.raises(KeyError) as exception:
        matrix.filter_by_range()
    assert msg in exception.value.args[0]


def test_sss_matrix_equality(sparse_array):
    def _create_array(name1, name2):
        matrix = StackedSparseScores(12, 10)
        matrix.add_coo_matrix(sparse_array1, name1)
        matrix.add_coo_matrix(sparse_array2, name2)
        return matrix
        
    sparse_array1 = coo_matrix(sparse_array.astype(np.int32))
    sparse_array[sparse_array % 10 == 0] = 0
    sparse_array = sparse_array/2
    sparse_array2 = coo_matrix(sparse_array.astype(np.int32))

    matrix = _create_array("scores1", "scores2")
    matrix_2 = _create_array("scores1", "scores2")
    assert matrix == matrix_2
    
    matrix_2.data[0] = 5
    assert matrix != matrix_2

    matrix_2 = _create_array("scores1", "scores_2")
    assert matrix != matrix_2

    matrix_2 = _create_array("scores1", "scores2")
    matrix_2.row[0] = 99
    assert matrix != matrix_2

    matrix_2 = _create_array("scores1", "scores2")
    matrix_2.col[0] = 99
    assert matrix != matrix_2


def test_sss_matrix_add_sparse_data(sparse_array):
    sparse_array = sparse_array[:5, :6]

    matrix = StackedSparseScores(5, 6)
    assert matrix.shape == (5, 6, 0)
    matrix.add_dense_matrix(sparse_array, "scoreA")
    assert matrix.shape == (5, 6, 1)
    assert np.all(matrix.data["scoreA"] == np.array([2, 10, 14, 22, 30, 34, 42]))

    # Add sparse scores
    new_scores = np.array([0.2, 0.5, 0.2, 0.1, 0.8, 1, 1])
    matrix.add_sparse_data(new_scores, "scoreB")
    assert np.all(matrix.to_array("scoreB")[:, 2] == np.array([0.2, 0., 0.1, 0., 1.]))
    assert matrix.to_array().shape == (5, 6)
    assert matrix.to_array()["scoreA"].shape == (5, 6)
    assert matrix.to_array()["scoreB"].shape == (5, 6)
    assert np.all(matrix.to_array()["scoreB"][:, 2] == np.array([0.2, 0., 0.1, 0., 1.]))
    assert np.all(matrix.to_array()["scoreA"][3, :] == np.array([30, 0, 0, 0, 34, 0]))


def test_sss_matrix_slicing():
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(arr, "test_score")

    # Test slicing
    assert matrix[0, 0] == 0
    assert matrix[2, 2] == 22
    assert matrix[2, 2, 0] == 22
    assert matrix[-1, -1] == 119

    # Slicing with [:]
    r, c, v = matrix[:, -1]
    assert np.all(v == np.arange(9, 120, 10))
    assert np.all(c == 9)
    r, _, v = matrix[2, :]
    r2, _, v2 = matrix[2]
    assert np.all(v == np.arange(20, 30))
    assert np.all(r == 2)
    assert np.all(v2 == np.arange(20, 30))
    assert np.all(r2 == 2)
    r, c, v = matrix["test_score"]
    r2, c2, v2 = matrix[:, :]
    r3, c3, v3 = matrix[:, :, 0]
    r4, c4, v4 = matrix[:, :, :]
    assert len(c) == len(c2) == len(c3) == len(c4) == 119
    assert len(r) == len(r2) == len(r3) == len(r4) == 119
    assert np.all(v == np.arange(1, 120))
    assert np.all(v2 == np.arange(1, 120))
    assert np.all(v3 == np.arange(1, 120))
    assert np.all(v4 == np.arange(1, 120))


def test_sss_matrix_slicing_mostly_empty_array():
    arr = np.zeros((4, 6))
    arr[3, 4] = 1.5
    matrix = StackedSparseScores(4, 6)
    matrix.add_dense_matrix(arr, "test_score")

    assert matrix[3, 3] == np.array([0])
    assert matrix[3, 4] == np.array([1.5])
    r, c, v = matrix[2, :]
    assert len(r) == len(c) == len(v) == 0
    r, c, v = matrix[3, :]
    assert (r, c, v) == (3, 4, 1.5)


@pytest.mark.parametrize("slicing_option", [
    "matrix[0, 1:3, 'scores1']",
    "matrix[:2, :, 0]",
    "matrix[:2, 1, 0]",
    "matrix[:, 1:, 0]",
    "matrix[1, 1:, 0]",
    "matrix[1, 1, :1]",
    "matrix[None]",
])
def test_sss_matrix_slicing_exceptions(sparse_array, slicing_option):
    msg = "Wrong slicing, or option not yet implemented"
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(sparse_array, "scores1")
    matrix.add_dense_matrix(sparse_array, "scores2")

    with pytest.raises(IndexError) as exception:
        exec(slicing_option)
    assert msg in exception.value.args[0]


def test_sss_matrix_filter_by_range():
    """Apply tresholds to really make the data sparse."""
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(arr, "test_score")
    matrix = matrix.filter_by_range(low=70, high=85)
    assert np.all(matrix.data["test_score"] == np.arange(71, 85))


def test_sss_matrix_filter_by_range_stacked():
    """Apply 2 sequential addition and filtering steps."""
    scores1 = np.arange(0, 120).reshape(12, 10)
    scores2 = np.arange(0, 120).reshape(12, 10).astype(float)
    scores2[scores2 < 80] = 0
    scores2[scores2 > 0] = 0.9

    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(scores1.astype(np.int64), "scores1")
    matrix = matrix.filter_by_range(low=70, high=85)
    assert matrix.shape == (12, 10, 1)
    assert np.all(matrix.data["scores1"] == np.arange(71, 85))

    matrix.add_dense_matrix(scores2, "scores2")
    matrix = matrix.filter_by_range("scores2", low=0.5)
    # Test slicing
    assert matrix[8, 1, 0] == np.array([81])
    assert matrix[8, 1, 1] == np.array([0.9])
    assert np.all(matrix[8, 3] == np.array([(83, 0.9)],
                                           dtype=[('scores1', '<i8'), ('scores2', '<f8')]))
    assert np.all(matrix[8, :][2]["scores1"] == np.array([80, 81, 82, 83, 84], dtype=np.int64))
    assert np.all(matrix[8, :][2]["scores2"] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))
    assert np.all(matrix[8, :, "scores2"][2] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))
    assert np.all(matrix[8, :, 1][2] == np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64))

    # Test more properties
    assert np.all(matrix.data["scores1"] == np.arange(80, 85))
    assert np.all(matrix.col == np.arange(0, 5))
    assert np.all(matrix.row == 8)
    assert matrix.shape == (12, 10, 2)
    assert matrix.score_names == ('scores1', 'scores2')


@pytest.mark.parametrize("input_index, msg", [
    [5, "out of range"],
    [-5, "out of range"],
])
def test_asindices(input_index, msg):
    array = StackedSparseScores(1, 1)
    with pytest.raises(IndexError) as exception:
        _ = array[input_index]
    assert msg in exception.value.args[0]


def test_missing_score_name():
    matrix = StackedSparseScores(2, 4)
    msg = "Array is empty."
    with pytest.raises(KeyError) as exception:
        _ = matrix.guess_score_name()
    assert msg in exception.value.args[0]
