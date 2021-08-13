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
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    adducts_dict = load_adducts_dict()

    if spectrum.get("parent_mass", None) is None or overwrite_existing_entry:
        parent_mass = None
        charge = spectrum.get("charge")
        adduct = clean_adduct(spectrum.get("adduct"))
        precursor_mz = spectrum.get("precursor_mz", None)
        if precursor_mz is None:
            print("Missing precursor m/z to derive parent mass.")
            return spectrum

        if estimate_from_adduct and adduct in adducts_dict:
            multiplier = adducts_dict[adduct]["mass_multiplier"]
            correction_mass = adducts_dict[adduct]["correction_mass"]
            parent_mass = precursor_mz * multiplier - correction_mass

        if parent_mass is None and charge is not None and charge != 0:
            # Otherwise assume adduct of shape [M+xH] or [M-xH]
            protons_mass = PROTON_MASS * charge
            precursor_mass = precursor_mz * abs(charge)
            parent_mass = precursor_mass - protons_mass

        if parent_mass is None:
            print("Not sufficient spectrum metadata to derive parent mass.")
        else:
            spectrum.set("parent_mass", float(parent_mass))
    return spectrum
