import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import FingerprintSimilarityParallel


def test_fingerprint_similarity_parallel_cosine():
    """Test cosine score matrix with known values."""
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

    similarity_measure = FingerprintSimilarityParallel(similarity_measure="cosine")
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix[1:, 1:] == pytest.approx(numpy.array([[1., 0.84515425],
                                                              [0.84515425, 1.]]), 0.001), "Expected different values."
    assert numpy.all(numpy.isnan(score_matrix[:, 0])), "Expected 'nan' entries."
    assert numpy.all(numpy.isnan(score_matrix[0, :])), "Expected 'nan' entries."


def test_fingerprint_similarity_parallel_jaccard():
    """Test jaccard index matrix with known values."""
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

    similarity_measure = FingerprintSimilarityParallel(similarity_measure="jaccard")
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix[1:, 1:] == pytest.approx(numpy.array([[1., 0.71428571],
                                                              [0.71428571, 1.]]), 0.001), "Expected different values."
    assert numpy.all(numpy.isnan(score_matrix[:, 0])), "Expected 'nan' entries."
    assert numpy.all(numpy.isnan(score_matrix[0, :])), "Expected 'nan' entries."


def test_fingerprint_similarity_parallel_dice():
    """Test dice score matrix with known values."""
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

    similarity_measure = FingerprintSimilarityParallel(similarity_measure="dice")
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix[1:, 1:] == pytest.approx(numpy.array([[1., 0.83333333],
                                                              [0.83333333, 1.]]), 0.001), "Expected different values."
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

    similarity_measure = FingerprintSimilarityParallel(set_empty_scores=0, similarity_measure="cosine")
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix == pytest.approx(numpy.array([[0, 0, 0],
                                                      [0, 1., 0.84515425],
                                                      [0, 0.84515425, 1.]]), 0.001), "Expected different values."
