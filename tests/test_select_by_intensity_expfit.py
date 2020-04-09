import numpy
from matchms import Spectrum
from matchms.filtering import select_by_intensity_expfit


def test_select_by_intensity_expfit():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([1, 10, 100, 1000], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=dict())

    select_by_intensity_expfit(spectrum, n_bins=10)


if __name__ == "__main__":
    test_select_by_intensity_expfit()
