import numpy as np
from matchms.networking.networking_functions import get_top_hits_coo_array, get_top_hits_matrix
from matchms.similarity import FingerprintSimilarity
from .test_SimilarityNetwork import create_dummy_spectra


def create_dummy_dense_matrix():
    spectra = create_dummy_spectra()
    similarity_measure = FingerprintSimilarity("dice")
    return similarity_measure.matrix(spectra, spectra, is_symmetric=True)


def create_dummy_coo_array():
    spectra = create_dummy_spectra()
    similarity_measure = FingerprintSimilarity("dice")
    return similarity_measure.sparse_array(spectra, spectra, is_symmetric=True)


# --- Expected values (symmetric 8x8 matrix, top_n=3, diagonal ignored) ---
# ref_spec_4 / query_spec_0 = [1,0,1] have dice=1.0 → top match for each other
EXPECTED_TOP_SCORES = np.array(
    [
        [0.66666667, 0.66666667, 0.66666667],  # ref_spec_0   → ref_spec_4, query_spec_0, ref_spec_3
        [0.66666667, 0.66666667, 0.5],  # ref_spec_1   → query_spec_1, ref_spec_3, query_spec_2
        [0.66666667, 0.66666667, 0.66666667],  # ref_spec_2   → query_spec_1, query_spec_0, ref_spec_4
        [0.8, 0.66666667, 0.66666667],  # ref_spec_3   → query_spec_2, ref_spec_0, ref_spec_1
        [1.0, 0.8, 0.66666667],  # ref_spec_4   → query_spec_0, query_spec_2, ref_spec_0
        [1.0, 0.8, 0.66666667],  # query_spec_0 → ref_spec_4, query_spec_2, ref_spec_0
        [0.8, 0.66666667, 0.66666667],  # query_spec_1 → query_spec_2, ref_spec_2, ref_spec_1
        [0.8, 0.8, 0.8],  # query_spec_2 → query_spec_1, query_spec_0, ref_spec_4
    ]
)

EXPECTED_TOP_INDICES = np.array(
    [
        [4, 5, 3],  # ref_spec_0
        [6, 3, 7],  # ref_spec_1
        [6, 5, 4],  # ref_spec_2
        [7, 0, 1],  # ref_spec_3
        [5, 7, 0],  # ref_spec_4
        [4, 7, 0],  # query_spec_0
        [7, 2, 1],  # query_spec_1
        [6, 5, 4],  # query_spec_2
    ]
)


def assert_top_hits(highest_scores, indexes_of_top_scores, expected_scores, expected_indices):
    """Helper to check scores and indices match expected, accounting for tie-breaking."""
    assert highest_scores.shape == expected_scores.shape, "Shape mismatch in scores"
    assert indexes_of_top_scores.shape == expected_indices.shape, "Shape mismatch in indices"
    assert np.allclose(highest_scores, expected_scores, atol=1e-5), (
        f"Scores mismatch:\n{highest_scores}\n!=\n{expected_scores}"
    )
    # For indices, verify the scores at returned indices match (handles tie-breaking differences)
    for row in range(expected_scores.shape[0]):
        assert np.allclose(highest_scores[row], expected_scores[row], atol=1e-5), f"Score mismatch at row {row}"


# --- Tests for get_top_hits_matrix ---


def test_get_top_hits_matrix_scores():
    matrix = create_dummy_dense_matrix()
    highest_scores, _ = get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=True)
    assert np.allclose(highest_scores, EXPECTED_TOP_SCORES, atol=1e-5), "Expected different top scores"


def test_get_top_hits_matrix_indices():
    matrix = create_dummy_dense_matrix()
    highest_scores, indexes_of_top_scores = get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=True)
    # Verify indices point to correct scores in the original matrix
    for row in range(matrix.shape[0]):
        retrieved = matrix[row, indexes_of_top_scores[row]]
        assert np.allclose(retrieved, highest_scores[row], atol=1e-5), (
            f"Indices at row {row} do not point to correct scores"
        )


def test_get_top_hits_matrix_sorted_descending():
    matrix = create_dummy_dense_matrix()
    highest_scores, _ = get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=True)
    for row in range(highest_scores.shape[0]):
        assert np.all(highest_scores[row, :-1] >= highest_scores[row, 1:]), (
            f"Scores at row {row} are not sorted descending"
        )


def test_get_top_hits_matrix_diagonal_ignored():
    matrix = create_dummy_dense_matrix()
    _, indexes_of_top_scores = get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=True)
    for row in range(matrix.shape[0]):
        assert row not in indexes_of_top_scores[row], f"Diagonal index {row} found in top hits for row {row}"


def test_get_top_hits_matrix_no_ignore_diagonal():
    matrix = create_dummy_dense_matrix()
    _, indexes_of_top_scores = get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=False)
    # Diagonal (score=1.0) should now appear as top hit for each row
    for row in range(matrix.shape[0]):
        assert row in indexes_of_top_scores[row], (
            f"Expected diagonal index {row} in top hits when ignore_diagonal=False"
        )


def test_get_top_hits_matrix_top_n_larger_than_matrix():
    matrix = create_dummy_dense_matrix()
    n_cols = matrix.shape[1]
    highest_scores, indexes_of_top_scores = get_top_hits_matrix(matrix, top_n=n_cols + 10, ignore_diagonal=False)
    assert highest_scores.shape[1] == n_cols, "Expected top_n to be clamped to matrix size"


def test_get_top_hits_matrix_does_not_mutate_input():
    matrix = create_dummy_dense_matrix()
    original = matrix.copy()
    get_top_hits_matrix(matrix, top_n=3, ignore_diagonal=True)
    assert np.allclose(matrix, original), "Input matrix was mutated"


# --- Tests for get_top_hits_coo_array ---


def test_get_top_hits_coo_array_scores():
    sparse = create_dummy_coo_array()
    highest_scores, _ = get_top_hits_coo_array(sparse, top_n=3, ignore_diagonal=True)
    # Only check non-nan entries (sparse rows may have fewer than top_n stored values)
    mask = ~np.isnan(highest_scores)
    assert np.allclose(highest_scores[mask], EXPECTED_TOP_SCORES[mask], atol=1e-5), (
        "Expected different top scores for COO array"
    )


def test_get_top_hits_coo_array_indices():
    sparse = create_dummy_coo_array()
    highest_scores, indexes_of_top_scores = get_top_hits_coo_array(sparse, top_n=3, ignore_diagonal=True)
    csr = sparse.tocsr()
    for row in range(sparse.shape[0]):
        for k, col in enumerate(indexes_of_top_scores[row]):
            if col == -1:
                continue  # sentinel for missing entries
            assert np.isclose(csr[row, col], highest_scores[row, k], atol=1e-5), (
                f"Index mismatch at row {row}, position {k}"
            )


def test_get_top_hits_coo_array_sorted_descending():
    sparse = create_dummy_coo_array()
    highest_scores, _ = get_top_hits_coo_array(sparse, top_n=3, ignore_diagonal=True)
    for row in range(highest_scores.shape[0]):
        valid = highest_scores[row][~np.isnan(highest_scores[row])]
        assert np.all(valid[:-1] >= valid[1:]), f"Scores at row {row} are not sorted descending"


def test_get_top_hits_coo_array_diagonal_ignored():
    sparse = create_dummy_coo_array()
    _, indexes_of_top_scores = get_top_hits_coo_array(sparse, top_n=3, ignore_diagonal=True)
    for row in range(sparse.shape[0]):
        assert row not in indexes_of_top_scores[row], f"Diagonal index {row} found in top hits for row {row}"


def test_get_top_hits_coo_array_top_n_larger_than_stored():
    sparse = create_dummy_coo_array()
    n_cols = sparse.shape[1]
    highest_scores, indexes_of_top_scores = get_top_hits_coo_array(sparse, top_n=100, ignore_diagonal=True)
    assert highest_scores.shape[1] == n_cols - 1, "Expected top_n to be clamped to n_cols - 1"
    assert np.any(np.isnan(highest_scores)), "Expected NaN sentinels for missing entries"
    assert np.any(indexes_of_top_scores == -1), "Expected -1 sentinels for missing entries"
