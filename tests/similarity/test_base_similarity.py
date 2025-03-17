import pytest

from matchms.similarity.BaseSimilarity import get_indexes_dense_matrix

@pytest.mark.parametrize("rows, columns",
                         [(2, 3),
                          (5, 6),])
def test_get_indexes_dense_matrix_not_symmetric(rows, columns):
    not_symmetric_rows, not_symmetric_columns = get_indexes_dense_matrix(rows, columns)
    assert len(not_symmetric_rows) == len(not_symmetric_columns)
    assert len(not_symmetric_rows) == rows*columns
    all_covered_positions = {(int(r), int(c)) for r, c in zip(not_symmetric_rows, not_symmetric_columns)}
    # Check that all coordinates are unique
    assert len(all_covered_positions) == len(not_symmetric_rows)
    all_possible_positions = {(r, c) for r in range(rows) for c in range(columns)}
    assert all_covered_positions == all_possible_positions

@pytest.mark.parametrize("rows, columns",
                         [(2, 2),
                          (5, 5), ])
def test_get_indexes_dense_matrix_symmetric(rows, columns):
    symmetric_rows, symmetric_columns = get_indexes_dense_matrix(rows, columns, is_symmetric=True)
    assert len(symmetric_rows) == len(symmetric_columns)

    assert len(symmetric_rows) == (rows*columns + rows)/2

    # check fo symmetric
    all_covered_positions = {(int(r), int(c)) for r, c in zip(symmetric_rows, symmetric_columns)}
    # Check that all coordinates are unique
    assert len(all_covered_positions) == len(symmetric_rows)
    all_possible_positions = {(r, c) for r in range(rows) for c in range(columns)}
    all_reversed_covered_positions = {(int(c), int(r)) for r, c in zip(symmetric_rows, symmetric_columns)}
    assert all_covered_positions | all_reversed_covered_positions == all_possible_positions
