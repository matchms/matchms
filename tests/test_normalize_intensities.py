from matchms import Spectrum
from matchms.filtering import normalize_intensities
import numpy


def test_normalize_intensities():

    mz = numpy.array([10, 20, 30, 40], dtype='float')
    intensities = numpy.array([0, 1, 10, 100], dtype='float')
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = normalize_intensities(spectrum_in)

    assert max(spectrum.peaks.intensities) == 1.0, "Expected the spectrum to be scaled to 1.0."
    assert numpy.array_equal(spectrum.peaks.mz, mz), "Expected the spectrum's intensity locations along "\
                                                     "the mz-axis to be unaffected."


def test_normalize_intensities_empty_peaks():

    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = normalize_intensities(spectrum_in)

    assert spectrum == spectrum_in, "Spectrum should remain unchanged."
