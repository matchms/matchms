import pytest
from .builder_spectrum import SpectrumBuilder


@pytest.mark.parametrize("input_dict, expected, export_style", [
    [{"precursor_mz": 101.01}, {"PrecursorMZ": 101.01}, "nist"],
    [{"precursormz": 101.01}, {"PRECURSORMZ": 101.01}, "riken"],
    [{"precursormz": 101.01}, {"PEPMASS": 101.01}, "gnps"],
    [{"charge": "2+"}, {"AC$MASS_SPECTROMETRY:CHARGE": "2+"}, "massbank"],
    [{"charge": -1}, {"Charge": -1}, "nist"],
    [{"ionmode": "Negative"}, {"IONMODE": "Negative"}, "riken"],
    [{"precursor_mz": 101.01, "retention_time": 12.5, "ms_level": "2"},
     {"PEPMASS": 101.01, "RTINSECONDS": 12.5, "MSLEVEL": "2"}, "sirius"],
    [{"adduct": "[M+H]+", "feature_id": "7", "compound_name": "Dummy"},
     {"ION": "[M+H]+", "FEATURE_ID": "7", "NAME": "Dummy"}, "sirius"],
    # SIRIUS reads the polarity from CHARGE and chokes on IONMODE values such as "n/a",
    # so ionmode is deliberately not exported in the sirius style.
    [{"charge": -1, "ionmode": "negative"}, {"CHARGE": -1}, "sirius"]])
def test_key_conversion(input_dict, expected, export_style):
    spectrum = SpectrumBuilder().with_metadata(input_dict).build()
    actual = spectrum.metadata_dict(export_style)

    assert actual == expected
