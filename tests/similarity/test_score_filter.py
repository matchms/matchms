import numpy as np
import pytest
from matchms.similarity.ScoreFilter import FilterScoreByValue


@pytest.mark.parametrize(
    "value, operator, score, expected",
    [[0.0, ">", np.array(1.0), True],
     [1.0, "<", np.array(3.0), False],
     [2.0, "<", np.array(1.0), True],
     [2.0, "==", np.array(1.0), False],
     [2.0, "!=", np.array(1.0), True],
     [1, "!=", np.array(1.0), False],
     [2.0, ">=", np.array(2.0), True],
     [2.0, "<=", np.array(2.0), True],
     [-1.0, "<", np.asarray((float(0), 0), dtype=[("score", np.float64), ("matches", "int")]), False],
     [-1.0, ">", np.asarray((float(0), 0), dtype=[("score", np.float64), ("matches", "int")]), True],
     ])
def test_filter_score_by_range(value, operator, score, expected):
    score_filter = FilterScoreByValue(value, operator)
    should_be_stored = score_filter.keep_score(np.array(score))
    assert expected == should_be_stored

@pytest.mark.parametrize(
    "value, operator, score, expected, score_name",
    [[1.0, ">", np.asarray((float(2.0), 0), dtype=[("score", np.float64), ("matches", "int")]), False, "matches"],
     [1.0, ">", np.asarray((float(2.0), 0), dtype=[("score", np.float64), ("matches", "int")]), True, "score"],
     ])
def test_filter_score_by_range_other_type(value, operator, score, expected, score_name):
    score_filter = FilterScoreByValue(value, operator, score_name)
    should_be_stored = score_filter.keep_score(np.array(score))
    assert expected == should_be_stored

@pytest.mark.parametrize(
    "value, operator, score, expected",
    [[0.0, ">", np.array([1.0, -1.0]), np.array([1.0, 0])],
     [0.0, "<", np.array([1.0, -1.0]), np.array([0.0, -1.0])],
     [0.0, "<", np.array([1, -1], dtype=np.int64), np.array([0, -1])],
     [0.0, "!=", np.array([1.0, -1.0]), np.array([1.0, -1.0])],
     [0.0, "!=", np.array([[1.0, -1.0], [1.0, -1.0],]),
      np.array([[1.0, -1.0], [1.0, -1.0],])],
     [0.5, "<", np.asarray([(1.0, 0), (0.0, 0)], dtype=[("score", np.float64), ("matches", np.int64)]),
      np.asarray([(0.0, 0), (0.0, 0)], dtype=[("score", np.float64), ("matches", np.int64)]),],
     ])
def test_filter_matrix(value, operator, score, expected):
    score_filter = FilterScoreByValue(value, operator)
    output_matrix = score_filter.filter_matrix(score)
    assert np.array_equal(expected, output_matrix)
