import numpy as np
import pandas as pd
from matchms.filtering.filter_utils.load_known_adducts import load_known_adduct_conversions, load_known_adducts


def test_load_adducts_dict():
    """Test if correct dict is imported."""
    known_adducts = load_known_adducts()
    assert isinstance(known_adducts, pd.DataFrame), "Expected a pandas dataframe"
    assert "[M+2H+Na]3+" in list(known_adducts["adduct"]), "Expected adduct to be in the dataframe"
    assert "[M+CH3COO]-" in list(known_adducts["adduct"]), "Expected adduct to be in dictionary"
    assert known_adducts.loc[known_adducts["adduct"] == "[M+2H+Na]3+", "charge"].values[0] == 3, "Expected different entry"
    assert np.all([(key[0] == "[") for key in known_adducts["adduct"]]), "Expected all keys to start with '['."
    assert known_adducts.loc[known_adducts["adduct"] == "[M]+", "charge"].values[0] == 1, "Expected different entry"


def test_load_known_adduct_conversions():
    """Test if correct data is imported."""
    adduct_conversions = load_known_adduct_conversions()
    assert isinstance(adduct_conversions, dict), "Expected result to be dict"
    assert adduct_conversions["[M-H-H2O]"] == "[M-H2O-H]-", "Expected different conversion rule."
