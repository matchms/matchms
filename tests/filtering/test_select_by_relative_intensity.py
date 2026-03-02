import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering import select_by_relative_intensity
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrum_in() -> Spectrum:
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([1, 10, 100, 1000], dtype="float")
    return SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()


@pytest.mark.parametrize(
    "intensity_from, intensity_to, expected_mz, expected_intensities",
    [
        [0, 1, np.array([10, 20, 30, 40], dtype="float"), np.array([1, 10, 100, 1000], dtype="float")],
        [0.01, 1, np.array([20, 30, 40], dtype="float"), np.array([10, 100, 1000], dtype="float")],
        [0, 0.99, np.array([10, 20, 30], dtype="float"), np.array([1, 10, 100], dtype="float")],
        [0.01, 0.99, np.array([20, 30], dtype="float"), np.array([10, 100], dtype="float")],
    ],
)
def test_select_by_relative_intensity(spectrum_in, intensity_from, intensity_to, expected_mz, expected_intensities):
    spectrum = select_by_relative_intensity(spectrum_in, intensity_from=intensity_from, intensity_to=intensity_to)

    assert spectrum.peaks.mz.size == len(expected_mz)
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert np.array_equal(spectrum.peaks.mz, expected_mz)
    assert np.array_equal(spectrum.peaks.intensities, expected_intensities)


def test_select_by_relative_intensity_with_from_parameter_too_small(spectrum_in: Spectrum):
    with pytest.raises(AssertionError):
        select_by_relative_intensity(spectrum_in, intensity_from=-10.0)


def test_select_by_relative_intensity_with_to_parameter_too_large(spectrum_in: Spectrum):
    with pytest.raises(AssertionError):
        select_by_relative_intensity(spectrum_in, intensity_to=10.0)


def test_select_by_relative_intensity_with_empty_peaks():
    """Within certain workflows it can happen that spectra are passed which
    have empty arrays as peaks. Functions shouldn't break in those cases."""
    spectrum_in = SpectrumBuilder().build()

    spectrum = select_by_relative_intensity(spectrum_in, intensity_from=0.01, intensity_to=0.99)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."
