from matchms import Spectrum
from matchms.filtering import require_minimum_number_of_peaks
import numpy


def test_require_minimum_number_of_peaks_no_params():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_number_of_peaks(spectrum_in)

    assert spectrum is None, "Expected None because the number of peaks (4) is less than the default threshold (10)."


def test_require_minimum_number_of_peaks_required_4():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_1_no_parent_mass():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_1():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=10)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_ratio_none():
    """Test if parent_mass scaling is properly ignored when not passing ratio_required."""
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=100)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_10():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=100)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (10)."


def test_require_minimum_number_of_peaks_required_5_or_1():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=10)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=5, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (5)."


def test_require_minimum_number_of_peaks_required_5_or_10():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=100)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=5, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (10)."
