import numpy as np
import pytest
from matchms.filtering import remove_peaks_around_precursor_mz
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrum_in():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    metadata = {"precursor_mz": 60.}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(
        intensities).with_metadata(metadata).build()
    return spectrum_in


def test_remove_peaks_around_precursor_mz_no_params(spectrum_in):
    """Using defaults with precursor mz present."""
    spectrum = remove_peaks_around_precursor_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_remove_peaks_around_precursor_mz_tolerance_20(spectrum_in):
    """Set mz_tolerance to 20."""
    spectrum = remove_peaks_around_precursor_mz(spectrum_in, mz_tolerance=20)

    assert len(spectrum.peaks) == 3, "Expected 3 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [10., 20., 30.], "Expected different peaks to remain."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 1.}).build()

    spectrum = remove_peaks_around_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_remove_peaks_around_precursor_without_precursor_mz(spectrum_in):
    """Test if correct assert error is raised for missing precursor-mz."""
    spectrum_in.metadata = {}

    with pytest.raises(AssertionError) as msg:
        _ = remove_peaks_around_precursor_mz(spectrum_in)

    assert str(msg.value) == "Precursor mz absent.", "Expected different error message."


def test_remove_peaks_around_precursor_with_wrong_precursor_mz(spectrum_in):
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in.set("precursor_mz", "445.0")

    with pytest.raises(AssertionError) as msg:
        _ = remove_peaks_around_precursor_mz(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = remove_peaks_around_precursor_mz(spectrum_in)
    assert spectrum is None
