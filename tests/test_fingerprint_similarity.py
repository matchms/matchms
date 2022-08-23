import numpy as np
import pytest
from matchms import Spectrum, calculate_scores
from matchms.similarity import FingerprintSimilarity


@pytest.mark.parametrize("test_method, expected_score", [("cosine", 0.6761234), ("jaccard", 0.5), ("dice", 2/3)])
def test_fingerprint_similarity_pair_calculations(test_method, expected_score):
    """Test cosine score pair with two fingerprint."""
    fingerprint1 = numpy.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = numpy.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(similarity_measure=test_method)
    score_pair = similarity_measure.pair(spectrum1, spectrum2)
    assert score_pair == pytest.approx(expected_score, 1e-6), "Expected different score."


@pytest.mark.parametrize("test_method", ["cosine", "jaccard", "dice"])
def test_fingerprint_similarity_parallel_empty_fingerprint(test_method):
    """Test score matrix with empty fingerprint using the provided methods."""
    fingerprint1 = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = numpy.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(similarity_measure=test_method)
    score_matrix = similarity_measure.matrix([spectrum1, spectrum2],
                                             [spectrum1, spectrum2])
    assert score_matrix == pytest.approx(numpy.array([[0, 0],
                                                      [0, 1.]]), 0.001), "Expected different values."


@pytest.mark.parametrize("test_method, expected_score", [("cosine", 0.84515425),
                                                         ("jaccard", 0.71428571),
                                                         ("dice", 0.83333333)])
def test_fingerprint_similarity_parallel(test_method, expected_score):
    """Test score matrix with known values for the provided methods."""
    spectrum0 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={})

    fingerprint1 = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = numpy.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(similarity_measure=test_method)
    score_matrix = similarity_measure.matrix([spectrum0, spectrum1, spectrum2],
                                             [spectrum0, spectrum1, spectrum2])
    expected_matrix = numpy.array([[1., expected_score],
                                   [expected_score, 1.]])
    assert score_matrix[1:, 1:] == pytest.approx(expected_matrix, 0.001), "Expected different values."
    assert numpy.all(numpy.isnan(score_matrix[:, 0])), "Expected 'nan' entries."
    assert numpy.all(numpy.isnan(score_matrix[0, :])), "Expected 'nan' entries."


def test_fingerprint_similarity_parallel_cosine_set_empty_to_0():
    """Test cosine score matrix with known values. Set non-exising values to 0."""
    spectrum0 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={})

    fingerprint1 = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = numpy.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(set_empty_scores=0, similarity_measure="cosine")
    score_matrix = similarity_measure.matrix([spectrum0, spectrum1, spectrum2],
                                             [spectrum0, spectrum1, spectrum2])
    assert score_matrix == pytest.approx(numpy.array([[0, 0, 0],
                                                      [0, 1., 0.84515425],
                                                      [0, 0.84515425, 1.]]), 0.001), "Expected different values."


def test_fingerprint_similarity_with_scores_sorting():
    """Test if score works with Scores.scores_by_query and sorting."""
    spectrum0 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={})

    fingerprint1 = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"fingerprint": fingerprint1})

    fingerprint2 = numpy.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    spectrum2 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"fingerprint": fingerprint2})

    similarity_measure = FingerprintSimilarity(set_empty_scores=0, similarity_measure="cosine")

    scores = calculate_scores([spectrum0, spectrum1, spectrum2],
                              [spectrum0, spectrum1, spectrum2],
                              similarity_measure)

    scores_by_ref_sorted = scores.scores_by_query(spectrum1, sort=True)
    expected_scores = numpy.array([1.0, 0.84515425])
    assert numpy.allclose(numpy.array([x[1] for x in scores_by_ref_sorted]), expected_scores, atol=1e-6), \
        "Expected different scores and/or order."
