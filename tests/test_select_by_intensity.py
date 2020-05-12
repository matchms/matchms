import numpy
from matchms import Spectrum
from matchms.filtering import select_by_intensity


def test_select_by_intensity_no_parameters_1():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_intensity(spectrum_in)

    assert spectrum.peaks.mz.size == 2
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([10, 100], dtype="float"))


def test_select_by_intensity_no_parameters_2():

    mz = numpy.array([998, 999, 1000, 1001, 1002], dtype="float")
    intensities = numpy.array([198, 199, 200, 201, 202], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_intensity(spectrum_in)

    assert spectrum.peaks.mz.size == 3
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([998, 999, 1000], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([198, 199, 200], dtype="float"))


def test_select_by_intensity_with_from_parameter():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_intensity(spectrum_in, intensity_from=15.0)

    assert spectrum.peaks.mz.size == 1
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([30], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([100], dtype="float"))


def test_select_by_intensity_with_to_parameter():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_intensity(spectrum_in, intensity_to=35.0)

    assert spectrum.peaks.mz.size == 1
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([20], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([10], dtype="float"))


def test_select_by_intensity_with_from_and_to_parameters():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = select_by_intensity(spectrum_in, intensity_from=15.0, intensity_to=135.0)

    assert spectrum.peaks.mz.size == 1
    assert spectrum.peaks.mz.size == spectrum.peaks.intensities.size
    assert numpy.array_equal(spectrum.peaks.mz, numpy.array([30], dtype="float"))
    assert numpy.array_equal(spectrum.peaks.intensities, numpy.array([100], dtype="float"))
