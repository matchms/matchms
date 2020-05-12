import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import CosineGreedy


def test_cosine_greedy_without_parameters():

    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190, 300, 490, 510, 1090], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy()
    score, n_matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.81421, 0.0001), "Expected different cosine score."
    assert n_matches == 3


def test_cosine_score_greedy_with_tolerance_0_2():
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([50, 100, 200, 299.5, 489.5, 510.5, 1040], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy(tolerance=0.2)
    score, n_matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.081966, 0.0001), "Expected different cosine score."
    assert n_matches == 2


def test_cosine_score_greedy_with_tolerance_2_0():

    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy(tolerance=2.0)
    score, n_matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.903412, 0.0001), "Expected different cosine score."
    assert n_matches == 6


def test_cosine_score_greedy_order_of_arguments():

    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)

    cosine_greedy = CosineGreedy(tolerance=2.0)
    score_1_2, n_matches_1_2 = cosine_greedy(norm_spectrum_1, norm_spectrum_2)
    score_2_1, n_matches_2_1 = cosine_greedy(norm_spectrum_2, norm_spectrum_1)

    assert score_1_2 == score_2_1, "Expected that the order of the arguments would not matter."
    assert n_matches_1_2 == n_matches_2_1, "Expected that the order of the arguments would not matter."
