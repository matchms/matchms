import numpy
from matchms import Spectrum
from matchms.filtering import remove_peaks_around_precursor_mz


def test_remove_peaks_around_precursor_mz_no_params():
    """Using defaults with precursor mz present."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 60.)

    spectrum = remove_peaks_around_precursor_mz(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_remove_peaks_around_precursor_mz_tolerance_20():
    """Set mz_tolerance to 20."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 60.)

    spectrum = remove_peaks_around_precursor_mz(spectrum_in, mz_tolerance=20)

    assert len(spectrum.peaks) == 3, "Expected 3 peaks to remain."
    assert spectrum.peaks.mz.tolist() == [10., 20., 30.], "Expected different peaks to remain."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    mz = numpy.array([], dtype="float")
    intensities = numpy.array([], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    spectrum_in.set("precursor_mz", 1.)

    spectrum = remove_peaks_around_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."
