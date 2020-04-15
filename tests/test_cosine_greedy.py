import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import CosineGreedy
from matchms.filtering import normalize_intensities


def test_cosine_greedy_without_parameters():

    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190, 300, 490, 510, 1090], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy")
    score, matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.81421, 0.0001), 'expected different cosine score'
    assert matches == 3, 'expected different number of matches'


def test_cosine_score_greedy_with_tolerance_0_2():
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy", tolerance=0.2)
    score, matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.2273, 0.0001), 'expected different cosine score'
    assert matches == 4, 'expected different number of matches'


def test_cosine_score_greedy_with_tolerance_1_0():

    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy", tolerance=1.0)
    score, matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)
    score_inverse, _ = cosine_greedy(norm_spectrum_2, norm_spectrum_1)

    assert score == pytest.approx(0.87121, 0.0001), 'expected different cosine score'
    assert score == pytest.approx(score_inverse, 0.0001), 'scores should be equal'
    assert matches == 5, 'expected different number of matches'


def test_cosine_score_greedy_with_tolerance_2_0():

    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy", tolerance=2.0)
    score, matches = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.903412, 0.0001), 'expected different cosine score'
    assert matches == 6, 'expected different number of matches'


if __name__ == '__main__':
    test_cosine_greedy_without_parameters()
    test_cosine_score_greedy_with_tolerance_0_2()
    test_cosine_score_greedy_with_tolerance_1_0()
    test_cosine_score_greedy_with_tolerance_2_0()
