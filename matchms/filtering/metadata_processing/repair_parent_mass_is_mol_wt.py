from matchms import Spectrum
from matchms.filtering.filters.repair_parent_mass_is_mol_wt import RepairParentMassIsMolWt


def repair_parent_mass_is_mol_wt(spectrum_in: Spectrum, mass_tolerance: float):
    """Changes the parent mass from molecular mass into monoistopic mass

    Manual entered precursor mz is sometimes wrongly added as Molar weight instead of monoisotopic mass
    """

    spectrum = RepairParentMassIsMolWt(mass_tolerance).process(spectrum_in)
    return spectrum
