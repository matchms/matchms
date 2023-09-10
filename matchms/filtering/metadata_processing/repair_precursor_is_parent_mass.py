from matchms import Spectrum
from matchms.filtering.filters.repair_precursor_is_parent_mass import RepairPrecursorIsParentMass


def repair_precursor_is_parent_mass(spectrum_in: Spectrum,
                                    mass_tolerance):
    """Repairs parent mass and precursor mz if the parent mass is entered instead of the precursor_mz"""

    spectrum = RepairPrecursorIsParentMass(mass_tolerance).process(spectrum_in)
    return spectrum
