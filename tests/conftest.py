import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_losses


@pytest.fixture
def mz():
    return numpy.array([10, 20, 30, 40], dtype='float')


@pytest.fixture
def intensities():
    return numpy.array([0, 1, 10, 100], dtype='float')


@pytest.fixture
def spectrum_without_losses(mz, intensities):
    return Spectrum(mz=mz, intensities=intensities)


@pytest.fixture
def spectrum_with_losses(mz, intensities):
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata={"precursor_mz": 45.0})
    return add_losses(spectrum)


@pytest.fixture
def spectrum_without_peaks():
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    return Spectrum(mz=mz, intensities=intensities)
