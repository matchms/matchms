import pytest
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "input_dict, expected, export_style",
    [
        [{"precursor_mz": 101.01}, {"PrecursorMZ": 101.01}, "nist"],
        [{"precursormz": 101.01}, {"PRECURSORMZ": 101.01}, "riken"],
        [{"precursormz": 101.01}, {"PEPMASS": 101.01}, "gnps"],
        [{"charge": "2+"}, {"AC$MASS_SPECTROMETRY:CHARGE": "2+"}, "massbank"],
        [{"charge": -1}, {"Charge": -1}, "nist"],
        [{"ionmode": "Negative"}, {"IONMODE": "Negative"}, "riken"],
    ],
)
def test_key_conversion(input_dict, expected, export_style):
    spectrum = SpectrumBuilder().with_metadata(input_dict).build()
    actual = spectrum.metadata_dict(export_style)

    assert actual == expected
