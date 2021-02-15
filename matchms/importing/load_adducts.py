import csv
import os
from functools import lru_cache
from typing import Dict
import numpy


@lru_cache(maxsize=4)
def load_adducts_dict() -> Dict[str, dict]:
    """Load dictionary of known adducts containing the adduct mass and charge.
    Makes sure that file loading is cached.

    Adduct information is based on information from
    https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator/
    and was extended by F.Huber and JJJ.v.d.Hooft.

    The full table can be found at
    https://github.com/matchms/matchms/blob/expand_adducts/matchms/data/known_adducts_table.csv

    TODO: change to relative path link or update link

    """
    known_adducts_file = os.path.join(os.path.dirname(__file__), "..", "data", "known_adducts_table.csv")
    assert os.path.isfile(known_adducts_file), "Could not find known_adducts_table.csv."

    with open(known_adducts_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        adducts_dict = dict()
        for row in reader:
            assert "adduct" in row
            adducts_dict[row["adduct"]] = {x[0]: x[1] for x in row.items() if x[0] != "adduct"}

    return _convert_and_fill_dict(adducts_dict)


@lru_cache(maxsize=4)
def load_known_adduct_conversions() -> Dict[str, dict]:
    """Load dictionary of known adduct conversions. Makes sure that file loading is cached.
    """
    adduct_conversions_file = os.path.join(os.path.dirname(__file__), "..", "data", "known_adduct_conversions.csv")
    assert os.path.isfile(adduct_conversions_file), "Could not find known_adduct_conversions.csv."

    with open(adduct_conversions_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        known_adduct_conversions = dict()
        for row in reader:
            known_adduct_conversions[row['input_adduct']] = row['corrected_adduct']

    return known_adduct_conversions


def _convert_and_fill_dict(adduct_dict: Dict[str, dict]) -> Dict[str, dict]:
    """Convert string entries to int/float and fill missing entries ('n/a')
    with best basic guesses."""
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_int(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    filled_dict = dict()
    for adduct, values in adduct_dict.items():
        ionmode = values["ionmode"]
        charge = values["charge"]
        mass_multiplier = values["mass_multiplier"]
        correction_mass = values["correction_mass"]

        if is_int(charge):
            charge = int(charge)
        else:
            charge = 1 * (ionmode == "positive") - 1 * (ionmode == "negative")
        values["charge"] = charge

        if is_float(mass_multiplier):
            mass_multiplier = float(mass_multiplier)
        else:
            mass_multiplier = 1.0
        values["mass_multiplier"] = mass_multiplier

        if is_float(correction_mass):
            correction_mass = float(correction_mass)
        else:
            correction_mass = 1.007276 * numpy.sign(charge)
        values["correction_mass"] = correction_mass

        filled_dict[adduct] = values
    return filled_dict
