import pytest
import pandas as pd
from matchms.importing import load_adducts_dict
from matchms.importing import load_adducts_table
from matchms.importing import load_known_adduct_conversions


def test_load_adducts_dict():
    """Test if correct dict is imported."""
    known_adducts = load_adducts_dict()
    assert isinstance(known_adducts["adducts_positive"], list), "Expected adducts_positive list"
    assert "[M+2H+Na]3+" in known_adducts["adducts_positive"], "Expected adduct to be in list"
    assert isinstance(known_adducts["adducts_negative"], list), "Expected adducts_negative list"
    assert "[M+CH3COO]-" in known_adducts["adducts_negative"], "Expected adduct to be in list"


def test_load_adducts_dict_no_file():
    """Test for not existing yaml filename."""
    known_adducts = load_adducts_dict(filename="nonexist.yaml")
    assert len(known_adducts["adducts_positive"]) == 0
    assert len(known_adducts["adducts_negative"]) == 0


def test_load_adducts_table():
    """Test if correct data is imported."""
    adduct_table = load_adducts_table()
    assert isinstance(adduct_table, pd.DataFrame), "Expected result to be dict"
    assert adduct_table.columns.to_list() == ['adduct', 'ionmode', 'charge',
                                              'mass_multiplier', 'correction_mass'], \
        "Expected different columns in adduct table"

def test_load_adducts_table_no_file():
    """Test if correct data is imported."""
    adduct_table = load_adducts_table(filename="nonexist.csv")
    assert adduct_table  is None, "Expected result to be None"


def test_load_known_adduct_conversions():
    """Test if correct data is imported."""
    adduct_conversions = load_known_adduct_conversions()
    assert isinstance(adduct_conversions, dict), "Expected result to be dict"
    assert adduct_conversions["[M-H-H2O]"] == "[M-H2O-H]-", "Expected different conversion rule."


def test_load_known_adduct_conversions_no_file():
    """Test if None is returned if file is missing."""
    adduct_conversions = load_known_adduct_conversions(filename="nonexist.json")
    assert adduct_conversions is None, "Expected result to be None"
