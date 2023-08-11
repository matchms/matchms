import csv
import os
from functools import lru_cache
from typing import Dict
import pandas as pd


@lru_cache(maxsize=4)
def load_known_adducts() -> pd.DataFrame:
    """Load dictionary of known adducts containing the adduct mass and charge.
    Makes sure that file loading is cached.

    Adduct information is based on information from
    https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator/
    and was extended by F.Huber and JJJ.v.d.Hooft.

    The full table can be found at
    https://github.com/matchms/matchms/blob/expand_adducts/matchms/data/known_adducts_table.csv

    """
    known_adducts_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "known_adducts_table.csv")
    assert os.path.isfile(known_adducts_file), "Could not find known_adducts_table.csv."

    with open(known_adducts_file, newline='', encoding='utf-8-sig') as csvfile:
        adducts_dataframe = pd.read_csv(csvfile)
        assert list(adducts_dataframe.columns) == ["adduct", "ionmode", "charge", "mass_multiplier", "correction_mass"]
    return adducts_dataframe


@lru_cache(maxsize=4)
def load_known_adduct_conversions() -> Dict[str, dict]:
    """Load dictionary of known adduct conversions. Makes sure that file loading is cached.
    """
    adduct_conversions_file = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                           "known_adduct_conversions.csv")
    assert os.path.isfile(adduct_conversions_file), "Could not find known_adduct_conversions.csv."

    with open(adduct_conversions_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        known_adduct_conversions = {}
        for row in reader:
            known_adduct_conversions[row['input_adduct']] = row['corrected_adduct']

    return known_adduct_conversions
