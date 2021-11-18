from ..constants import PROTON_MASS
from ..importing import load_adducts_dict
from ..typing import SpectrumType
from ..utils import clean_adduct


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
    def is_valid_charge(charge):
        return (charge is not None) and (charge != 0)

    def guess_charge_from_ionmode(spectrum):
        if spectrum.get('ionmode') == "positive":
            return 1
        if spectrum.get('ionmode') == "negative":
            return -1
        return 0

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    adducts_dict = load_adducts_dict()

    if spectrum.get("parent_mass", None) and not overwrite_existing_entry:
        return spectrum

    parent_mass = None
    charge = spectrum.get("charge")
    adduct = clean_adduct(spectrum.get("adduct"))
    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        print("Missing precursor m/z to derive parent mass.")
        return spectrum

    if estimate_from_adduct and (adduct in adducts_dict):
        multiplier = adducts_dict[adduct]["mass_multiplier"]
        correction_mass = adducts_dict[adduct]["correction_mass"]
        parent_mass = precursor_mz * multiplier - correction_mass

    if parent_mass is None:
        # Handle missing charge if ionmode is given
        charge_guess = guess_charge_from_ionmode(spectrum)
        if not is_valid_charge(charge) and is_valid_charge(charge_guess):
            charge = charge_guess
            print("Missing charge entry, but ionmode detected. "
                  "Consider prior run of `correct_charge()` filter.")

        if is_valid_charge(charge):
            # Assume adduct of shape [M+xH] or [M-xH]
            protons_mass = PROTON_MASS * charge
            precursor_mass = precursor_mz * abs(charge)
            parent_mass = precursor_mass - protons_mass

    if parent_mass is None:
        print("Not sufficient spectrum metadata to derive parent mass.")
    else:
        spectrum.set("parent_mass", float(parent_mass))
    return spectrum
