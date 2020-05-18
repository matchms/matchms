import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import ModifiedCosineNumba


def test_modified_cosine_numba_without_parameters():
    """Test without parameters and precursor-m/z. Should be default cosine."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190, 300, 490, 510, 1090], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineNumba()
    score, n_matches = modified_cosine(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.81421, 0.0001), "Expected different modified cosine score."
    assert n_matches == 3, "Expected 3 matching peaks."


def test_modified_cosine_numba_with_mass_shift_5():
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1005.0})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineNumba()
    score, n_matches = modified_cosine(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.081966, 0.0001), "Expected different cosine score."
    assert n_matches == 2, "Expected 2 matching peaks."


def test_modified_cosine_numba_with_mass_shift_5_tolerance_2():
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([105, 205, 305, 306, 505, 517], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata={"precursor_mz": 1005})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineNumba(tolerance=2.0)
    score, n_matches = modified_cosine(norm_spectrum_1, norm_spectrum_2)

    assert score == pytest.approx(0.903412, 0.0001), "Expected different modified cosine score."
    assert n_matches == 6, "Expected 6 matching peaks."


def test_cmodified_cosine_numba_order_of_input_spectrums():
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1005.0})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineNumba(tolerance=2.0)
    score_1_2, n_matches_1_2 = modified_cosine(norm_spectrum_1, norm_spectrum_2)
    score_2_1, n_matches_2_1 = modified_cosine(norm_spectrum_2, norm_spectrum_1)

    assert score_1_2 == score_2_1, "Expected that the order of the arguments would not matter."
    assert n_matches_1_2 == n_matches_2_1, "Expected that the order of the arguments would not matter."
