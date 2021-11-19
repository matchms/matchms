import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import require_minimum_number_of_peaks
from matchms.typing import SpectrumType


@pytest.fixture
def spectrum_in():
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    metadata = dict(parent_mass=10)
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)


def test_require_minimum_number_of_peaks_no_params(spectrum_in: SpectrumType):
    spectrum = require_minimum_number_of_peaks(spectrum_in)

    assert spectrum is None, "Expected None because the number of peaks (4) is less than the default threshold (10)."


def test_require_minimum_number_of_peaks_required_4(spectrum_in: SpectrumType):
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_1_no_parent_mass(spectrum_in: SpectrumType):
    spectrum_in.set("parent_mass", None)
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_1(spectrum_in: SpectrumType):
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_ratio_none(spectrum_in: SpectrumType):
    """Test if parent_mass scaling is properly ignored when not passing ratio_required."""
    spectrum_in.set("parent_mass", 100)

    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_require_minimum_number_of_peaks_required_4_or_10(spectrum_in: SpectrumType):
    spectrum_in.set("parent_mass", 100)
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (10)."


def test_require_minimum_number_of_peaks_required_5_or_1(spectrum_in: SpectrumType):
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=5, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (5)."


def test_require_minimum_number_of_peaks_required_5_or_10(spectrum_in: SpectrumType):
    spectrum_in.set("parent_mass", 100)
    spectrum = require_minimum_number_of_peaks(spectrum_in, n_required=5, ratio_required=0.1)

    assert spectrum is None, "Did not expect the spectrum to qualify because the number of peaks (4) is less " \
                             "than the required number (10)."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = require_minimum_number_of_peaks(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
