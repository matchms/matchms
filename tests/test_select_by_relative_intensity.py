from matchms import Spectrum
from matchms.filtering import select_by_relative_intensity
import numpy
import pytest


def test_select_by_relative_intensity_no_parameters():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_relative_intensity(spectrum_in)

    assert spectrum.peaks.mz.size == 4
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([10, 20, 30, 40], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([1, 10, 100, 1000], dtype="float"))


def test_select_by_relative_intensity_with_from_parameter():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_relative_intensity(spectrum_in, intensity_from=0.01)

    assert spectrum.peaks.mz.size == 3
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([20, 30, 40], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([10, 100, 1000], dtype="float"))


def test_select_by_relative_intensity_with_from_parameter_too_small():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    with pytest.raises(AssertionError):
        select_by_relative_intensity(spectrum_in, intensity_from=-10.0)


def test_select_by_relative_intensity_with_to_parameter():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_relative_intensity(spectrum_in, intensity_to=0.99)

    assert spectrum.peaks.mz.size == 3
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([10, 20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([1, 10, 100], dtype="float"))


def test_select_by_relative_intensity_with_to_parameter_too_large():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    with pytest.raises(AssertionError):
        select_by_relative_intensity(spectrum_in, intensity_to=10.0)


def test_select_by_relative_intensity_with_from_and_to_parameters():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_relative_intensity(spectrum_in, intensity_from=0.01, intensity_to=0.99)

    assert spectrum.peaks.mz.size == 2
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([10, 100], dtype="float"))


def test_select_by_relative_intensity_with_empty_peaks():
    """Within certain workflows it can happen that spectrums are passed which
    have empty arrays as peaks. Functions shouldn't break in those cases."""
    mz = numpy.array([], dtype="float")
    intensities = numpy.array([], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_relative_intensity(spectrum_in, intensity_from=0.01, intensity_to=0.99)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."
