import numpy
import pytest
from matchms.filtering import add_precursor_mz
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, expected", [
    [{"precursor_mz": 444.0}, 444.0],
    [{}, None],
    [{"pepmass": (444.0, 10)}, 444.0]
])
def test_add_precursor_mz(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


@pytest.mark.parametrize("key, value, expected", [
    ["precursor_mz", "444.0", 444.0],
    ["precursormz", "15.6", 15.6],
    ["precursormz", 15.0, 15.0],
    ["precursor_mass", "17.887654", 17.887654],
    ["precursor_mass", "N/A", None],
    ["precursor_mass", "test", None],
    ["pepmass", (33.89, 50), 33.89],
    ["pepmass", "None", None],
    ["pepmass", None, None]])
def test_add_precursor_mz_no_precursor_mz(key, value, expected):
    """Test if precursor_mz is correctly derived if "precursor_mz" is str."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {key: value}
    spectrum_in = SpectrumBuilder().with_metadata(
        metadata).with_mz(mz).with_intensities(intensities).build()

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == expected, "Expected different precursor_mz."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
