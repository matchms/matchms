import numpy as np
import pytest
from matchms.filtering import interpret_pepmass
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("input_pepmass, expected_results",
                         [((None), (None, None, None)),
                          ((896.05), (896.05, None, None)),
                          ((896.05, None), (896.05, None, None)),
                          ((896.05, 1111.2, "2-"), (896.05, 1111.2, -2)),
                          ((896.05, 1111.2, "2+"), (896.05, 1111.2, 2)),
                          ((896.05, 1111.2, -1), (896.05, 1111.2, -1))])
def test_interpret_pepmass(input_pepmass, expected_results):
    """Test if example inputs are correctly converted"""
    mz = np.array([100, 200.])
    intensities = np.array([0.7, 0.1])
    metadata = {'pepmass': input_pepmass}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == expected_results, \
        "Expected different 3 values."


def test_interpret_pepmass_charge_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.])
    intensities = np.array([0.7, 0.1])
    metadata = {'pepmass': (896.05, 1111.2, "2-"),
                'charge': -1}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (896.05, 1111.2, -2), \
        "Expected different 3 values."
    assert "Overwriting existing charge -1 with new one: -2" in caplog.text, \
        "Expected different log message"


def test_interpret_pepmass_mz_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.])
    intensities = np.array([0.7, 0.1])
    metadata = {'pepmass': (203, 44, "2-"),
                'precursor_mz': 202}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (203, 44, -2), \
        "Expected different 3 values."
    assert "Overwriting existing precursor_mz 202 with new one: 203" in caplog.text, \
        "Expected different log message"


def test_interpret_pepmass_intensity_present(caplog):
    """Test if example inputs are correctly converted when entries already exist"""
    mz = np.array([100, 200.])
    intensities = np.array([0.7, 0.1])
    metadata = {'pepmass': (203, 44, "2-"), 'precursor_intensity': 100}
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum = interpret_pepmass(spectrum_in)
    mz = spectrum.get("precursor_mz")
    intensity = spectrum.get("precursor_intensity")
    charge = spectrum.get("charge")
    assert (mz, intensity, charge) == (203, 44, -2), \
        "Expected different 3 values."
    assert "Overwriting existing precursor_intensity 100 with new one: 44" in caplog.text, \
        "Expected different log message"


def test_empty_spectrum():
    spectrum_in = None
    spectrum = interpret_pepmass(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
