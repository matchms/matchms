import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import FingerprintSimilarityParallel


def test_fingerprint_similarity_parallel_no_param():

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

    similarity_measure = FingerprintSimilarityParallel()
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix[1:, 1:] == pytest.approx(numpy.array([[1., 0.84515425],
                                                              [0.84515425, 1.]]), 0.001), "Expected different values."
    assert numpy.all(numpy.isnan(score_matrix[:, 0])), "Expected 'nan' entries."
    assert numpy.all(numpy.isnan(score_matrix[0, :])), "Expected 'nan' entries."


def test_fingerprint_similarity_parallel_set_empty_to_0():

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

    similarity_measure = FingerprintSimilarityParallel(set_empty_scores=0)
    score_matrix = similarity_measure([spectrum0, spectrum1, spectrum2],
                                      [spectrum0, spectrum1, spectrum2])
    assert score_matrix == pytest.approx(numpy.array([[0, 0, 0],
                                                      [0, 1., 0.84515425],
                                                      [0, 0.84515425, 1.]]), 0.001), "Expected different values."
