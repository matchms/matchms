import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import ModifiedCosine


def test_modified_cosine_without_precursor_mz():
    """Test without precursor-m/z. Should raise assertion error."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190, 300, 490, 510, 1090], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"))

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine()

    with pytest.raises(AssertionError) as msg:
        modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    expected_message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
    assert str(msg.value) == expected_message


def test_modified_cosine_with_mass_shift_5():
    """Test modified cosine on two spectra with mass set shift."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1005.0})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine()
    score = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    assert score["score"] == pytest.approx(0.081966, 0.0001), "Expected different cosine score."
    assert score["matches"] == 2, "Expected 2 matching peaks."


def test_modified_cosine_with_mass_shift_5_tolerance_2():
    """Test modified cosine on two spectra with mass set shift and tolerance."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 200, 20, 100], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([105, 205, 305, 306, 505, 517], dtype="float"),
                          intensities=numpy.array([10, 10, 500, 100, 20, 100], dtype="float"),
                          metadata={"precursor_mz": 1005})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine(tolerance=2.0)
    score = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    assert score["score"] == pytest.approx(0.96788, 0.0001), "Expected different modified cosine score."
    assert score["matches"] == 6, "Expected 6 matching peaks."


def test_modified_cosine_with_mass_shifted_and_unshifted_matches():
    """Test modified cosine on two spectra with mass set shift.
    In this example 5 peak pairs are possible, but only 3 should be selected (every peak
    can only be counted once!)"""
    spectrum_1 = Spectrum(mz=numpy.array([100, 110, 200, 300, 400, 500, 600], dtype="float"),
                          intensities=numpy.array([100, 50, 1, 80, 1, 1, 50], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([110, 200, 300, 310, 700, 800], dtype="float"),
                          intensities=numpy.array([100, 1, 90, 90, 1, 100], dtype="float"),
                          metadata={"precursor_mz": 1010.0})

    modified_cosine = ModifiedCosine()
    score = modified_cosine.pair(spectrum_1, spectrum_2)
    spec1 = spectrum_1.peaks.intensities
    spec2 = spectrum_2.peaks.intensities
    peak_pairs_multiplied = spec1[0] * spec2[0] + spec1[3] * spec2[3] + spec1[2] * spec2[1]
    expected_score = peak_pairs_multiplied / numpy.sqrt(numpy.sum(spec1 ** 2) * numpy.sum(spec2 ** 2))
    assert score["score"] == pytest.approx(expected_score, 0.00001), "Expected different cosine score."
    assert score["matches"] == 3, "Expected 3 matching peaks."


def test_modified_cosine_order_of_input_spectrums():
    """Test modified cosine on two spectra in changing order."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"),
                          intensities=numpy.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1005.0})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine(tolerance=2.0)
    score_1_2 = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)
    score_2_1 = modified_cosine.pair(norm_spectrum_2, norm_spectrum_1)

    assert score_1_2["score"] == score_2_1["score"], "Expected that the order of the arguments would not matter."
    assert score_1_2 == score_2_1, "Expected that the order of the arguments would not matter."


def test_modified_cosine_with_mass_shift_5_no_matches_expected():
    """Test modified cosine on two spectra with no expected matches."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                          intensities=numpy.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([120, 220, 320], dtype="float"),
                          intensities=numpy.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": 1005})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine(tolerance=1.0)
    score = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    assert score["score"] == pytest.approx(0.0, 1e-5), "Expected different modified cosine score."
    assert score["matches"] == 0, "Expected 0 matching peaks."


def test_modified_cosine_precursor_mz_as_string():
    """Test modified cosine on two spectra with precursor_mz given as string."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                          intensities=numpy.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=numpy.array([120, 220, 320], dtype="float"),
                          intensities=numpy.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": "1005.0"})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosine(tolerance=1.0)
    with pytest.raises(AssertionError) as msg:
        _ = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    expected_message = "Precursor_mz must be of type int or float. Apply 'add_precursor_mz' filter first."
    assert str(msg.value) == expected_message
