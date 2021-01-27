from ..constants import PROTON_MASS
from ..importing import load_adducts_table
from ..typing import SpectrumType


def add_parent_mass(spectrum_in: SpectrumType, estimate_from_adduct: bool = True) -> SpectrumType:
    """Add parentmass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor m/z together
    with charge and/or adduct. Will take precursor m/z from either "precursor_mz"
    or "pepmass" field.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    estimate_from_adduct
        When set to True, use adduct to estimate actual molecular mass ("parent mass").
        Default is True. Still switches back to charge-based estimate if adduct does not match
        known adduct.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    adducts_table = load_adducts_table()

    if spectrum.get("parent_mass", None) is None:
        parent_mass = None
        charge = spectrum.get("charge")
        adduct = spectrum.get("adduct")
        print(charge, adduct)
        # Assert if sufficent metadata is present
        # TODO: maybe also accept charge=0 ?
        assert charge != 0 or adduct is not None, "Not sufficient spectrum metadata to derive parent mass."
        if not estimate_from_adduct:
            assert charge != 0, "Not sufficient spectrum metadata to derive parent mass."
        try:
            precursor_mz = spectrum.get("precursor_mz", None)
            if precursor_mz is None:
                precursor_mz = spectrum.get("pepmass")[0]
        except KeyError:
            print("Not sufficient spectrum metadata to derive parent mass.")

        spectrum = spectrum_in.clone()
        if estimate_from_adduct and adduct is not None:
            if adduct in adducts_table.adduct.to_list():
                adduct_data = adducts_table[adducts_table.adduct == adduct]
                multiplier = adduct_data.mass_multiplier.values
                correction_mass = adduct_data.correction_mass.values
                parent_mass = precursor_mz * multiplier - correction_mass
                print(parent_mass)
        
        if parent_mass is None:
            # Otherwise assume adduct of shape M+xH or M-xH
            protons_mass = PROTON_MASS * charge
            precursor_mass = precursor_mz * abs(charge)
            parent_mass = precursor_mass - protons_mass

        if parent_mass is not None:
            spectrum.set("parent_mass", parent_mass)
    return spectrum
