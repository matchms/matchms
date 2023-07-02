import csv
import logging
import os
import re
from functools import lru_cache
from typing import Dict
import pandas as pd


logger = logging.getLogger("matchms")


def clean_adduct(spectrum_in):
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    spectrum_in
        Matchms Spectrum object.
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()
    adduct = spectrum.get("adduct")

    cleaned_adduct = _clean_adduct(adduct)
    if adduct != cleaned_adduct:
        spectrum.set("adduct", cleaned_adduct)
        logger.info("The adduct %d was set to %s", adduct, cleaned_adduct)
    return spectrum


def _clean_adduct(adduct: str) -> str:
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
