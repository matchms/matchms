from matchms import Spectrum
from matchms.filtering import select_by_intensity
import numpy


def test_select_by_intensity_no_bounds():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity(spectrum)

    assert len(spectrum.mz) == 2
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([10, 100], dtype="float"))


def test_select_by_intensity_lower_bound_only():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity(spectrum, intensity_from=15.0)

    assert len(spectrum.mz) == 1
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([30], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([100], dtype="float"))


def test_select_by_intensity_upper_bound_only():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity(spectrum, intensity_to=35.0)

    assert len(spectrum.mz) == 1
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([20], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([10], dtype="float"))


def test_select_by_intensity_lower_and_upper_bound():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity(spectrum, intensity_from=15.0, intensity_to=135.0)

    assert len(spectrum.mz) == 1
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([30], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([100], dtype="float"))


def test_select_by_intensity_no_bounds_2():

    mz = numpy.array([998, 999, 1000, 1001, 1002], dtype="float")
    intensities = numpy.array([198, 199, 200, 201, 202], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity(spectrum)

    assert len(spectrum.mz) == 3
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([998, 999, 1000], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([198, 199, 200], dtype="float"))


if __name__ == '__main__':
    test_select_by_intensity_lower_bound_only()
    test_select_by_intensity_upper_bound_only()
    test_select_by_intensity_lower_and_upper_bound()
    test_select_by_intensity_no_bounds()
    test_select_by_intensity_no_bounds_2()
