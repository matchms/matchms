from matchms import Spectrum
from matchms.filtering.filters.repair_adduct_based_on_smiles import RepairAdductBasedOnSmiles


def repair_adduct_based_on_smiles(spectrum_in: Spectrum,
                                  mass_tolerance,
                                  accept_parent_mass_is_mol_wt):
    """If the parent mass is wrong due to a wrong of is derived from the precursor mz
    To do this the charge and adduct are used"""

    spectrum = RepairAdductBasedOnSmiles(mass_tolerance, accept_parent_mass_is_mol_wt).process(spectrum_in)
    return spectrum
