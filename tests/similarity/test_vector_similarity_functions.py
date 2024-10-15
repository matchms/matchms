"""Test for vector similarity functions. Will run test on both numba compiled
and fully python-based versions of functions."""
import numpy as np
import pytest
from matchms.similarity.vector_similarity_functions import (
    cosine_similarity, cosine_similarity_matrix, dice_similarity,
    dice_similarity_matrix, jaccard_index, jaccard_similarity_matrix,
    ruzicka_similarity, ruzicka_similarity_matrix)


def test_cosine_similarity_compiled():
    """Test cosine similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = cosine_similarity(vector1, vector1)
    score12 = cosine_similarity(vector1, vector2)
    score22 = cosine_similarity(vector2, vector2)

    assert score12 == 2 / np.sqrt(2 * 4), "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_cosine_similarity_all_zeros_compiled():
    """Test cosine similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = cosine_similarity(vector1, vector1)
    score12 = cosine_similarity(vector1, vector2)
    score22 = cosine_similarity(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_cosine_similarity_matrix_compiled():
    """Test cosine similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]])

    scores = cosine_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_cosine_similarity():
    """Test cosine similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = cosine_similarity.py_func(vector1, vector1)
    score12 = cosine_similarity.py_func(vector1, vector2)
    score22 = cosine_similarity.py_func(vector2, vector2)

    assert score12 == 2 / np.sqrt(2 * 4), "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_cosine_similarity_all_zeros():
    """Test cosine similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = cosine_similarity.py_func(vector1, vector1)
    score12 = cosine_similarity.py_func(vector1, vector2)
    score22 = cosine_similarity.py_func(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_cosine_similarity_matrix():
    """Test cosine similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]])

    scores = cosine_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_dice_similarity_compiled():
    """Test dice similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = dice_similarity(vector1, vector1)
    score12 = dice_similarity(vector1, vector2)
    score22 = dice_similarity(vector2, vector2)

    assert score12 == 2 * 2/6, "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_dice_similarity_all_zeros_compiled():
    """Test dice similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = dice_similarity(vector1, vector1)
    score12 = dice_similarity(vector1, vector2)
    score22 = dice_similarity(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_dice_similarity_matrix_compiled():
    """Test dice similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [1, 0, 1, 1]])

    scores = dice_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.4],
                                   [0.5, 0.8]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_dice_similarity():
    """Test dice similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = dice_similarity.py_func(vector1, vector1)
    score12 = dice_similarity.py_func(vector1, vector2)
    score22 = dice_similarity.py_func(vector2, vector2)

    assert score12 == 2 * 2/6, "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_dice_similarity_all_zeros():
    """Test dice similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = dice_similarity.py_func(vector1, vector1)
    score12 = dice_similarity.py_func(vector1, vector2)
    score22 = dice_similarity.py_func(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_dice_similarity_matrix():
    """Test dice similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [1, 0, 1, 1]])

    scores = dice_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[0.5, 0.4],
                                   [0.5, 0.8]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_jaccard_index_compiled():
    """Test jaccard similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = jaccard_index(vector1, vector1)
    score12 = jaccard_index(vector1, vector2)
    score22 = jaccard_index(vector2, vector2)

    assert score12 == 2 / 4, "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_jaccard_index_all_zeros_compiled():
    """Test jaccard similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = jaccard_index(vector1, vector1)
    score12 = jaccard_index(vector1, vector2)
    score22 = jaccard_index(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_jaccard_similarity_matrix_compiled():
    """Test jaccard similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [1, 0, 1, 1]])

    scores = jaccard_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[1/3, 1/4],
                                   [1/3, 2/3]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_jaccard_index():
    """Test jaccard similarity score calculation."""
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = jaccard_index.py_func(vector1, vector1)
    score12 = jaccard_index.py_func(vector1, vector2)
    score22 = jaccard_index.py_func(vector2, vector2)

    assert score12 == 2 / 4, "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


def test_jaccard_index_all_zeros():
    """Test jaccard similarity score calculation with empty vector."""
    vector1 = np.array([0, 0, 0, 0])
    vector2 = np.array([1, 1, 1, 1])
    score11 = jaccard_index.py_func(vector1, vector1)
    score12 = jaccard_index.py_func(vector1, vector2)
    score22 = jaccard_index.py_func(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


def test_jaccard_similarity_matrix():
    """Test jaccard similarity scores calculation."""
    vectors1 = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])
    vectors2 = np.array([[0, 1, 1, 0],
                            [1, 0, 1, 1]])

    scores = jaccard_similarity_matrix(vectors1, vectors2)
    expected_scores = np.array([[1/3, 1/4],
                                   [1/3, 2/3]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


def test_ruzicka_similarity():
    """Test ruzicka similarity score calculation."""
    vector1 = np.array([1.0, 1.0, 0, 0])
    vector2 = np.array([1.0, 1.0, 1.0, 1.0])
    vector3 = np.array([0, 0, 0, 0])
    vector4 = np.array([0, 0])
    weights = np.array([0.5, 1.0, 2.0, 0.5])

    score11 = ruzicka_similarity.py_func(vector1, vector1)
    score12 = ruzicka_similarity.py_func(vector1, vector2)
    score22 = ruzicka_similarity.py_func(vector2, vector2)
    score12w = ruzicka_similarity(vector1, vector2, weights)
    score33 = ruzicka_similarity(vector3, vector3)

    assert score11 == score22 == 1.0, "Expected identity score."

    # Different vectors
    expected_diff_vectors = (1 + 1 + 0 + 0) / (1 + 1 + 1 + 1)
    assert pytest.approx(score12) == expected_diff_vectors, "Expected different score."

    # Using weights
    min_sum = 0.5 * 1 + 1 * 1 + 2 * 0 + 0.5 * 0
    max_sum = 0.5 * 1 + 1 * 1 + 2 * 1 + 0.5 * 1
    expected_diff_vectors_weighting = min_sum / max_sum
    assert pytest.approx(score12w) == expected_diff_vectors_weighting, f"Expected weighted similarity: {expected_diff_vectors_weighting}"

    assert score33 == 0.0, "Zero vectors should return a similarity of 0.0"

    # Test diff vector sizes
    with pytest.raises(ValueError):
        ruzicka_similarity(vector1, vector4)


def test_ruzicka_similarity_matrix():
    """Test ruzicka similarity scores calculation."""
    reference1 = np.array([[1, 2, 0], [1, 1, 0], [0, 1.5, 2]])
    reference2 = np.array([[1, 2, 0], [1, 2.1, 0], [1, 1, 0], [0, 1.5, 2]])
    query = np.array([[4, 1, 1], [4, 1, 0], [0.5, 1, 1]])
    weights = np.array([0.5, 1.0, 1.0])

    scores_no_weights = ruzicka_similarity_matrix(reference1, query)
    expected_scores_no_weights = np.array([
        [0.28571429, 0.33333333, 0.375],
        [0.33333333, 0.4, 0.5],
        [0.26666667, 0.13333333, 0.5]
    ])
    assert scores_no_weights == pytest.approx(expected_scores_no_weights, 1e-7), "Expected different scores."

    scores = ruzicka_similarity_matrix(reference2, query, weights)
    expected_scores = np.array([
        [0.3, 0.375, 0.35714286],
        [0.29411765, 0.36585366, 0.34722222],
        [0.375, 0.5, 0.5],
        [0.36363636, 0.18181818, 0.53333333]
    ])
    assert scores.shape == (reference2.shape[0], query.shape[0]), "Expected different score shape"
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."
