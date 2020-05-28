import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import CosineHungarian


def test_cosine_hungarian_without_parameters():
    """Test if example with default parameters will give expected results."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190, 300, 490, 510, 1090], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_hungarian = CosineHungarian()
    score, n_matches = cosine_hungarian(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.81421, 0.0001), "Expected different cosine score."
    assert n_matches == 3, "Expected different number of matching peaks."


def test_cosine_hungarian_with_tolerance_0_2():
    """Test if example with tolerance=0.2 will give expected results."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([50, 100, 200, 299.5, 489.5, 510.5, 1040], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_hungarian = CosineHungarian(tolerance=0.2)
    score, n_matches = cosine_hungarian(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.081966, 0.0001), "Expected different cosine score."
    assert n_matches == 2, "Expected different number of matching peaks."


def test_cosine_hungarian_with_tolerance_2_0():
    """Test if example with tolerance=2.0 will give expected results."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    cosine_hungarian = CosineHungarian(tolerance=2.0)
    score, n_matches = cosine_hungarian(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.903412, 0.0001), "Expected different cosine score."
    assert n_matches == 6, "Expected different number of matching peaks."


def test_cosine_hungarian_order_of_arguments():
    """Test if score(A,B) == score(B,A)."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata=dict())

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 300, 301, 500, 512], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata=dict())

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)

    cosine_hungarian = CosineHungarian(tolerance=2.0)
    score_1_2, n_matches_1_2 = cosine_hungarian(norm_spectrum_1, norm_spectrum_2)
    score_2_1, n_matches_2_1 = cosine_hungarian(norm_spectrum_2, norm_spectrum_1)

    assert score_1_2 == score_2_1, "Expected that the order of the arguments would not matter."
    assert n_matches_1_2 == n_matches_2_1, "Expected that the order of the arguments would not matter."


def test_cosine_hungarian_case_where_greedy_would_fail():
    """Test case that would fail for cosine greedy implementations."""
    spectrum_1 = Spectrum(mz=numpy.array([100.005, 100.016]),
                          intensities=numpy.array([1.0, 0.9]),
                          metadata={})

    spectrum_2 = Spectrum(mz=numpy.array([100.005, 100.01]),
                          intensities=numpy.array([0.9, 1.0]),
                          metadata={})

    cosine_hungarian = CosineHungarian(tolerance=0.01)
    score, n_matches = cosine_hungarian(spectrum_1, spectrum_2)
    assert score == pytest.approx(0.994475, 0.0001), "Expected different cosine score."
    assert n_matches == 2, "Expected different number of matching peaks."
