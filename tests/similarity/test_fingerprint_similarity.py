import numpy as np
import pytest
from sparsestack import StackedSparseArray
from matchms import Spectrum, calculate_scores
from matchms.similarity import FingerprintSimilarity


@pytest.mark.parametrize("test_method, expected_score", [("cosine", 0.6761234), ("jaccard", 0.5), ("dice", 2/3)])
def test_fingerprint_similarity_pair_calculations(test_method, expected_score):
    """Test cosine score pair with two fingerprint."""
    fingerprint1 = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(similarity_measure=test_method)
    score_pair = similarity_measure.pair(spectrum1, spectrum2)
    assert score_pair == pytest.approx(expected_score, 1e-6), "Expected different score."


@pytest.mark.parametrize("test_method", ["cosine", "jaccard", "dice"])
def test_fingerprint_similarity_parallel_empty_fingerprint(test_method):
    """Test score matrix with empty fingerprint using the provided methods."""
    fingerprint1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(similarity_measure=test_method)
    score_matrix = similarity_measure.matrix([spectrum1, spectrum2],
                                             [spectrum1, spectrum2])
    assert score_matrix == pytest.approx(np.array([[0, 0],
                                                      [0, 1.]]), 0.001), "Expected different values."


@pytest.mark.parametrize("test_method, expected_score, set_empty", [("cosine", 0.84515425, np.nan),
                                                         ("jaccard", 0.71428571, np.nan),
                                                         ("dice", 0.83333333, 0)])
def test_fingerprint_similarity_parallel(test_method, expected_score, set_empty):
    """Test score matrix with known values for the provided methods."""
    spectrum0 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={})

    fingerprint1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=np.array([], dtype="float"),
                         intensities=np.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(set_empty_scores=set_empty, similarity_measure=test_method)
    score_matrix = similarity_measure.matrix([spectrum0, spectrum1, spectrum2],
                                             [spectrum0, spectrum1, spectrum2])
    expected_matrix = np.array([[set_empty, set_empty, set_empty],
                                   [set_empty, 1, expected_score],
                                   [set_empty, expected_score, 1]])
    if isinstance(score_matrix, (StackedSparseArray)):
        score_matrix = score_matrix.to_array()
    assert np.allclose(score_matrix , expected_matrix, equal_nan=True) , "Expected different values."


def test_fingerprint_similarity_with_scores_sorting():
    """Test if score works with Scores.scores_by_query and sorting."""
    spectrum0 = Spectrum(mz=np.array([100.0, 101.0], dtype="float"),
                         intensities=np.array([0.4, 0.5], dtype="float"),
                         metadata={})

    fingerprint1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=np.array([100.0, 101.0], dtype="float"),
                         intensities=np.array([0.4, 0.5], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=np.array([100.0, 101.0], dtype="float"),
                         intensities=np.array([0.4, 0.5], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(set_empty_scores=0, similarity_measure="cosine")

    scores = calculate_scores([spectrum0, spectrum1, spectrum2],
                              [spectrum0, spectrum1, spectrum2],
                              similarity_measure)

    scores_by_ref_sorted = scores.scores_by_query(spectrum1, sort=True)
    expected_scores = np.array([1.0, 0.84515425])
    assert np.allclose(np.array([x[1] for x in scores_by_ref_sorted]), expected_scores, atol=1e-6), \
        "Expected different scores and/or order."
