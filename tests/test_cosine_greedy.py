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
    score = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.81421, 0.0001), 'expected different cosine score'


def test_cosine_score_greedy_with_tolerance_0_2():
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([50, 100, 200, 299.5, 489.5, 510.5, 1040], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy", tolerance=0.2)
    score = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.081966, 0.0001), 'expected different cosine score'


def test_cosine_score_greedy_with_tolerance_0_5():

    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([50, 100, 200, 299.5, 489.5, 510.5, 1040], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_greedy = CosineGreedy("cosine-greedy", tolerance=0.5)
    score = cosine_greedy(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.6284203, 0.0001), 'expected different cosine score'


if __name__ == '__main__':
    test_cosine_greedy_without_parameters()
    test_cosine_score_greedy_with_tolerance_0_2()
    test_cosine_score_greedy_with_tolerance_0_5()
