import numpy
from matchms.importing import load_adducts_dict
from matchms.importing import load_known_adduct_conversions
from matchms.importing.load_adducts import _convert_and_fill_dict


def test_load_adducts_dict():
    """Test if correct dict is imported."""
    known_adducts = load_adducts_dict()
    assert isinstance(known_adducts, dict), "Expected dictionary"
    assert "[M+2H+Na]3+" in known_adducts, "Expected adduct to be in dictionary"
    assert "[M+CH3COO]-" in known_adducts, "Expected adduct to be in dictionary"
    assert known_adducts["[M+2H+Na]3+"]["charge"] == 3, "Expected different entry"
    assert numpy.all([(key[0] == "[") for key in known_adducts]), \
        "Expected all keys to start with '['."
    assert known_adducts["[M]+"]["charge"] == 1, "Expected different added entry"
    assert numpy.allclose(known_adducts["[M]-"]["correction_mass"], -1.007276, atol=1e-5), \
        "Expected different added entry"


def test_convert_and_fill_dict():
    """Test if conversion is done right."""
    test_dict = dict()
    test_dict["[M]+"] = {'ionmode': 'negative',
                         'charge': "n/a",
                         'mass_multiplier': "n/a",
                         'correction_mass': "n/a"}
    converted_dict = _convert_and_fill_dict(test_dict)
    assert isinstance(converted_dict["[M]+"], dict), "Expected dictionary"
    assert converted_dict["[M]+"]["charge"] == -1, "Expected charge of -1"
    assert converted_dict["[M]+"]["mass_multiplier"] == 1, "Expected mass_multiplier of 1"
    assert numpy.allclose(converted_dict["[M]+"]["correction_mass"], -1.007276, atol=1e-5), \
        "Expected correction_mass to be changed to -1.007276"


def test_load_known_adduct_conversions():
    """Test if correct data is imported."""
    adduct_conversions = load_known_adduct_conversions()
    assert isinstance(adduct_conversions, dict), "Expected result to be dict"
    assert adduct_conversions["[M-H-H2O]"] == "[M-H2O-H]-", "Expected different conversion rule."
