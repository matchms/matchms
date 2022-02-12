import numpy as np
from matchms.StackedSparseScores import StackedSparseScores


def test_sss_matrix():
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    assert matrix.shape == (12, 10, 0)
    matrix.add_dense_matrix(arr, "test_score")
    assert matrix.shape == (12, 10, 1)
    assert np.all(matrix.data["test_score"] == np.arange(1, 120))
    assert matrix.data.get("other_name") is None


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
    r, c, v = matrix[2, :]
    assert np.all(v == np.arange(20, 30))
    assert np.all(r == 2)
    r, c, v = matrix["test_score"]
    r2, c2, v2 = matrix[:, :]
    assert len(c) == len(c2) == 119
    assert len(r) == len(r2) == 119
    assert np.all(v == np.arange(1, 120))
    assert np.all(v2 == np.arange(1, 120))


def test_sss_matrix_filter_by_range():
    """Apply tresholds to really make the data sparse."""
    arr = np.arange(0, 120).reshape(12, 10)
    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(arr, "test_score")
    matrix.filter_by_range(low=70, high=85)
    assert np.all(matrix.data["test_score"] == np.arange(71, 85))


def test_sss_matrix_filter_by_range_stacked():
    """Apply 2 sequential addition and filtering steps."""
    scores1 = np.arange(0, 120).reshape(12, 10)
    scores2 = np.arange(0, 120).reshape(12, 10).astype(float)
    scores2[scores2 < 80] = 0
    scores2[scores2 > 0] = 0.9

    matrix = StackedSparseScores(12, 10)
    matrix.add_dense_matrix(scores1, "scores1")
    matrix.filter_by_range(low=70, high=85)
    assert matrix.shape == (12, 10, 1)
    assert np.all(matrix.data["scores1"] == np.arange(71, 85))

    matrix.add_dense_matrix(scores2, "scores2")
    matrix.filter_by_range("scores2", low=0.5)
    assert matrix.shape == (12, 10, 2)
    assert matrix[8, 1, 0] == np.array([81])
    assert matrix[8, 1, 1] == np.array([0.9])
    assert np.all(matrix.data["scores1"] == np.arange(80, 85))
    assert np.all(matrix.col == np.arange(0, 5))
    assert np.all(matrix.row == 8)
    assert matrix.score_names == ['scores1', 'scores2']
