import csv
import json
import os
from functools import lru_cache
from typing import Dict
import numpy


@lru_cache(maxsize=4)
def load_adducts_dict() -> Dict:
    """Load dictionary of known adducts containing the adduct mass and charge.
    Makes sure that file loading is cached.

    Adduct information is based on information from
    https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator/
    and was extended by F.Huber and JJJ.v.d.Hooft.
    """
    def convert_if_possible(entry, expected_type=float):
        try:
            entry = expected_type(entry)
        except ValueError:
            pass
        return entry

    def convert_and_fill_dict(adduct_dict):
        """Convert string entries to int/float and fill missing entries ('n/a')
        with best basic guesses."""
        filled_dict = dict()
        for adduct, values in adduct_dict.items():
            ionmode = values["ionmode"]
            charge = convert_if_possible(values["charge"], int)
            mass_multiplier = convert_if_possible(values["mass_multiplier"], float)
            correction_mass = convert_if_possible(values["correction_mass"], float)

            if not isinstance(charge, int):
                charge = 1 * (ionmode == "positive") - 1 * (ionmode == "negative")
            values["charge"] = charge

            if not isinstance(mass_multiplier, float):
                mass_multiplier = 1.0
            values["mass_multiplier"] = mass_multiplier

            if not isinstance(correction_mass, float):
                correction_mass = 1.007276 * numpy.sign(charge)
            values["correction_mass"] = correction_mass

            filled_dict[adduct] = values
        return filled_dict

    known_adducts_file = os.path.join(os.path.dirname(__file__), "..", "data", "known_adducts_table.csv")

    if not os.path.isfile(known_adducts_file):
        print("Could not find .csv file with known adducts.")
        return None

    with open(known_adducts_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        adducts_dict = dict()
        for row in reader:
            assert "adduct" in row
            adducts_dict[row["adduct"]] = {x[0]: x[1] for x in row.items() if x[0] != "adduct"}

    return convert_and_fill_dict(adducts_dict)


@lru_cache(maxsize=4)
def load_known_adduct_conversions() -> Dict:
    """Load dictionary of known adduct conversions. Makes sure that file loading is cached.
    """
    adduct_conversions_file = os.path.join(os.path.dirname(__file__), "..", "data", "known_adduct_conversions.json")

    if not os.path.isfile(adduct_conversions_file):
        print("Could not find json file with known adduct conversions.")
        return None

    known_adduct_conversions = json.load(open(adduct_conversions_file, "r"))

    return known_adduct_conversions
