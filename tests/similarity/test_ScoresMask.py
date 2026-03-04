import numpy as np
import pytest
from scipy.sparse import coo_array
from matchms.similarity.ScoresMask import ScoresMask


# --- from_matrix tests ---


@pytest.mark.parametrize(
    "value, operator, score, expected_rows, expected_cols",
    [
        [0.0, ">", np.array([[1.0, -1.0]]), [0], [0]],
        [0.0, "<", np.array([[1.0, -1.0]]), [0], [1]],
        [0.0, ">", np.array([[1.0, -1.0], [1.0, -1.0]]), [0, 1], [0, 0]],
        [0.0, "<", np.array([[1.0, -1.0], [1.0, -1.0]]), [0, 1], [1, 1]],
        [0.0, "<", np.array([[1, -1]], dtype=np.int64), [0], [1]],
        [0.0, "!=", np.array([[1.0, -1.0]]), [0, 0], [0, 1]],
    ],
)
def test_from_matrix(value, operator, score, expected_rows, expected_cols):
    mask = ScoresMask.from_matrix(score, operator, value)
    assert np.array_equal(expected_rows, mask.idx_row)
    assert np.array_equal(expected_cols, mask.idx_col)


def test_from_matrix_no_matches():
    """Test that an empty mask is returned when no values match."""
    score = np.array([[0.5, 0.3], [0.2, 0.1]])
    mask = ScoresMask.from_matrix(score, ">", 0.9)
    assert len(mask) == 0
    assert mask.idx_row.shape == (0,)
    assert mask.idx_col.shape == (0,)


def test_from_matrix_all_match():
    """Test that all indices are returned when all values match."""
    score = np.array([[0.5, 0.3], [0.2, 0.1]])
    mask = ScoresMask.from_matrix(score, ">", 0.0)
    assert len(mask) == 4


def test_from_matrix_includes_zeros():
    """Test that explicit zeros are included when condition covers 0.0."""
    score = np.array([[0.0, 0.5], [0.3, 0.0]])
    mask = ScoresMask.from_matrix(score, ">=", 0.0)
    assert len(mask) == 4


# --- from_coo_array tests ---
@pytest.mark.parametrize(
    "value, operator, expected_rows, expected_cols",
    [
        [0.8, ">", [0, 1, 1], [1, 0, 2]],
        [0.8, ">=", [0, 1, 1], [1, 0, 2]],
        [0.5, ">", [0, 1, 1, 2], [1, 0, 2, 1]],
        [0.85, "==", [1], [0]],
    ],
)
def test_from_coo_array(value, operator, expected_rows, expected_cols):
    arr = coo_array(np.array([[0.5, 0.9, 0.3], [0.85, 0.2, 0.95], [0.1, 0.75, 0.1]]))
    mask = ScoresMask.from_coo_array(arr, operator, value)
    assert np.array_equal(expected_rows, mask.idx_row)
    assert np.array_equal(expected_cols, mask.idx_col)


def test_from_coo_array_no_matches():
    """Test that an empty mask is returned when no values match."""
    arr = coo_array(np.array([[0.5, 0.3]]))
    mask = ScoresMask.from_coo_array(arr, ">", 0.9)
    assert len(mask) == 0


def test_from_coo_array_warns_on_zero_condition(caplog):
    """Test that a warning is raised when condition would include 0.0."""
    arr = coo_array(np.array([[0.0, 0.5], [0.3, 0.0]]))
    with caplog.at_level("WARNING", logger="matchms"):
        ScoresMask.from_coo_array(arr, ">=", 0.0)
    assert "0.0 values" in caplog.text


def test_from_coo_array_no_warning_above_zero(caplog):
    """Test that no warning is raised when condition excludes 0.0."""
    arr = coo_array(np.array([[0.5, 0.9]]))
    with caplog.at_level("WARNING", logger="matchms"):
        ScoresMask.from_coo_array(arr, ">", 0.0)
    assert "0.0 values" not in caplog.text


# --- ScoresMask general tests ---


def test_len():
    mask = ScoresMask(np.array([0, 1, 2]), np.array([1, 2, 3]), ncols=4, nrows=4)
    assert len(mask) == 3


def test_getitem_int():
    mask = ScoresMask(np.array([0, 1, 2]), np.array([3, 4, 5]), ncols=6, nrows=3)
    row, col = mask[1]
    assert row == 1 and col == 4


def test_getitem_slice():
    mask = ScoresMask(np.array([0, 1, 2]), np.array([3, 4, 5]), ncols=6, nrows=3)
    sliced = mask[1:]
    assert isinstance(sliced, ScoresMask)
    assert np.array_equal(sliced.idx_row, [1, 2])
    assert np.array_equal(sliced.idx_col, [4, 5])


def test_iter():
    mask = ScoresMask(np.array([0, 1, 2]), np.array([3, 4, 5]), ncols=6, nrows=3)
    pairs = list(mask)
    assert pairs == [(0, 3), (1, 4), (2, 5)]
