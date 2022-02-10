import logging
from matchms.filtering.load_adducts import load_adducts_dict
from ..constants import PROTON_MASS
from ..metadata_utils import clean_adduct
from ..typing import SpectrumType
from ..utils import get_first_common_element


logger = logging.getLogger("matchms")


_default_key = "parent_mass"
_accepted_keys = ["parentmass", "exact_mass"]
_accepted_types = (float, str, int)
_accepted_missing_entries = ["", "N/A", "NA", "n/a"]


def add_parent_mass(spectrum_in: SpectrumType, estimate_from_adduct: bool = True,
                    overwrite_existing_entry: bool = False) -> SpectrumType:
    """Add estimated parent mass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor m/z together
    with charge and/or adduct. Will take precursor m/z from "precursor_mz"
    as provided by running `add_precursor_mz`.
    For estimate_from_adduct=True this function will estimate the parent mass based on
    the mass and charge of known adducts. The table of known adduct properties can be
    found under :download:`matchms/data/known_adducts_table.csv </../matchms/data/known_adducts_table.csv>`.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    estimate_from_adduct
        When set to True, use adduct to estimate actual molecular mass ("parent mass").
        Default is True. Switches back to charge-based estimate if adduct does not match
        a known adduct.
    overwrite_existing_entry
        Default is False. If set to True, a newly computed value will replace existing ones.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    adducts_dict = load_adducts_dict()

    parent_mass = _get_parent_mass(spectrum.metadata)
    if parent_mass is not None and not overwrite_existing_entry:
        spectrum.set("parent_mass", parent_mass)
        return spectrum

    parent_mass = None
    charge = _get_charge(spectrum)
    adduct = clean_adduct(spectrum.get("adduct"))
    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        logger.warning("Missing precursor m/z to derive parent mass.")
        return spectrum

    if estimate_from_adduct and (adduct in adducts_dict):
        multiplier = adducts_dict[adduct]["mass_multiplier"]
        correction_mass = adducts_dict[adduct]["correction_mass"]
        parent_mass = precursor_mz * multiplier - correction_mass

    if (parent_mass is None) and _is_valid_charge(charge):
        # Assume adduct of shape [M+xH] or [M-xH]
        protons_mass = PROTON_MASS * charge
        precursor_mass = precursor_mz * abs(charge)
        parent_mass = precursor_mass - protons_mass

    if parent_mass is None:
        logger.warning("Not sufficient spectrum metadata to derive parent mass.")
    else:
        spectrum.set("parent_mass", float(parent_mass))
    return spectrum


def _get_parent_mass(metadata):
    parent_mass_key = get_first_common_element([_default_key] + _accepted_keys,
                                               metadata.keys())
    parent_mass = metadata.get(parent_mass_key)
    parent_mass = _convert_entry_to_num(parent_mass)
    if parent_mass not in _accepted_missing_entries:
        return parent_mass
    return None


def _convert_entry_to_num(entry):
    """Convert precursor_mz to number if possible. Otherwise return None."""
    if entry is None:
        return None
    if isinstance(entry, str) and entry in _accepted_missing_entries:
        return None
    if not isinstance(entry, _accepted_types):
        logger.warning("Found parent_mass of undefined type.")
        return None
    if isinstance(entry, str):
        try:
            return float(entry.strip())
        except ValueError:
            logger.warning("%s can't be converted to float.", entry)
            return None
    return entry


def _is_valid_charge(charge):
    return (charge is not None) and (charge != 0)


def _get_charge(spectrum):
    """Get charge from `Spectrum()` object.
    In case no valid charge is found, guess +1 or -1 based on ionmode.
    Else return 0.
    """
    charge = spectrum.get("charge")
    if _is_valid_charge(charge):
        return charge
    if spectrum.get('ionmode') == "positive":
        logger.info(
            "Missing charge entry, but positive ionmode detected. "
            "Consider prior run of `correct_charge()` filter.")
        return 1
    if spectrum.get('ionmode') == "negative":
        logger.info(
            "Missing charge entry, but negative ionmode detected. "
            "Consider prior run of `correct_charge()` filter.")
        return -1

    logger.warning(
        "Missing charge and ionmode entries. "
        "Consider prior run of `derive_ionmode()` and `correct_charge()` filters.")
    return 0
