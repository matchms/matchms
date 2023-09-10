from matchms.typing import SpectrumType
from matchms.filtering.filters.add_parent_mass import AddParentMass


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

    spectrum = AddParentMass(estimate_from_adduct, overwrite_existing_entry).process(spectrum_in)
    return spectrum
