from matchms import Spectrum
from matchms.filtering import select_by_mz
import numpy


def test_select_by_mz_no_bounds():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_mz(spectrum)

    assert len(spectrum.mz) == 4
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([10, 20, 30, 40], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([1, 10, 100, 1000], dtype="float"))


def test_select_by_mz_lower_bound_only():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_mz(spectrum, mz_from=15.0)

    assert len(spectrum.mz) == 3
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([20, 30, 40], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([10, 100, 1000], dtype="float"))


def test_select_by_mz_upper_bound_only():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_mz(spectrum, mz_to=35.0)

    assert len(spectrum.mz) == 3
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([10, 20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([1, 10, 100], dtype="float"))


def test_select_by_mz_lower_and_upper_bound():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_mz(spectrum, mz_from=15.0, mz_to=35.0)

    assert len(spectrum.mz) == 2
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([20, 30], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([10, 100], dtype="float"))


def test_select_by_mz_no_bounds_2():

    mz = numpy.array([998, 999, 1000, 1001, 1002], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000, 10000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_mz(spectrum)

    assert len(spectrum.mz) == 3
    assert len(spectrum.mz) == len(spectrum.intensities)
    assert numpy.array_equal(spectrum.mz, numpy.array([998, 999, 1000], dtype="float"))
    assert numpy.array_equal(spectrum.intensities, numpy.array([1, 10, 100], dtype="float"))


if __name__ == '__main__':
    test_select_by_mz_lower_bound_only()
    test_select_by_mz_upper_bound_only()
    test_select_by_mz_lower_and_upper_bound()
    test_select_by_mz_no_bounds()
    test_select_by_mz_no_bounds_2()
