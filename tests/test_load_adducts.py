import pandas as pd
from matchms.importing import load_adducts_dict
from matchms.importing import load_known_adduct_conversions


def test_load_adducts_dict():
    """Test if correct dict is imported."""
    known_adducts = load_adducts_dict()
    assert isinstance(known_adducts, dict), "Expected dictionary"
    assert "[M+2H+Na]3+" in known_adducts, "Expected adduct to be in dictionary"
    assert "[M+CH3COO]-" in known_adducts, "Expected adduct to be in dictionary"
    assert known_adducts["[M+CH3COO]-"]["charge"] == 3, "Expected different entry"
    assert numpy.all([(x[0] == "[") for x in known_adducts.keys()]), \
        "Expected all keys to start with '['."


def test_load_known_adduct_conversions():
    """Test if correct data is imported."""
    adduct_conversions = load_known_adduct_conversions()
    assert isinstance(adduct_conversions, dict), "Expected result to be dict"
    assert adduct_conversions["[M-H-H2O]"] == "[M-H2O-H]-", "Expected different conversion rule."
