import numpy
from matchms.importing import load_adducts_dict
from matchms.importing import load_known_adduct_conversions


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


def test_load_known_adduct_conversions():
    """Test if correct data is imported."""
    adduct_conversions = load_known_adduct_conversions()
    assert isinstance(adduct_conversions, dict), "Expected result to be dict"
    assert adduct_conversions["[M-H-H2O]"] == "[M-H2O-H]-", "Expected different conversion rule."
