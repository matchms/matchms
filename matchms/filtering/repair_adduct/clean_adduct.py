import csv
import os
import re
from functools import lru_cache
from typing import Dict


def clean_adduct(adduct: str) -> str:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    adduct
        Input adduct string to be cleaned/edited.
    """
    def get_adduct_charge(adduct):
        regex_charges = r"[1-3]{0,1}[+-]{1,2}$"
        match = re.search(regex_charges, adduct)
        if match:
            return match.group(0)
        return match

    def adduct_conversion(adduct):
        """Convert adduct if conversion rule is known"""
        adduct_conversions = load_known_adduct_conversions()
        if adduct in adduct_conversions:
            return adduct_conversions[adduct]
        return adduct

    if not isinstance(adduct, str):
        return adduct

    adduct = adduct.strip().replace("\n", "").replace("*", "")
    adduct = adduct.replace("++", "2+").replace("--", "2-")
    if adduct.startswith("["):
        return adduct_conversion(adduct)

    if adduct.endswith("]"):
        return adduct_conversion("[" + adduct)

    adduct_core = "[" + adduct
    # Remove parts that can confuse the charge extraction
    for mol_part in ["CH2", "CH3", "NH3", "NH4", "O2"]:
        if mol_part in adduct:
            adduct = adduct.split(mol_part)[-1]
    adduct_charge = get_adduct_charge(adduct)

    if adduct_charge is None:
        return adduct_conversion(adduct_core + "]")

    adduct_cleaned = adduct_core[:-len(adduct_charge)] + "]" + adduct_charge
    return adduct_conversion(adduct_cleaned)


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
    known_adducts_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "known_adducts_table.csv")
    assert os.path.isfile(known_adducts_file), "Could not find known_adducts_table.csv."

    with open(known_adducts_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        adducts_dict = {}
        for row in reader:
            assert "adduct" in row
            adducts_dict[row["adduct"]] = {x[0]: x[1] for x in row.items() if x[0] != "adduct"}

    formatted_adduct_dict = {}
    for adduct, values in adducts_dict.items():
        values["charge"] = int(values["charge"])
        values["mass_multiplier"] = float(values["mass_multiplier"])
        values["correction_mass"] = float(values["correction_mass"])
        formatted_adduct_dict[adduct] = values
    return formatted_adduct_dict


@lru_cache(maxsize=4)
def load_known_adduct_conversions() -> Dict[str, dict]:
    """Load dictionary of known adduct conversions. Makes sure that file loading is cached.
    """
    adduct_conversions_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "known_adduct_conversions.csv")
    assert os.path.isfile(adduct_conversions_file), "Could not find known_adduct_conversions.csv."

    with open(adduct_conversions_file, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        known_adduct_conversions = {}
        for row in reader:
            known_adduct_conversions[row['input_adduct']] = row['corrected_adduct']

    return known_adduct_conversions

