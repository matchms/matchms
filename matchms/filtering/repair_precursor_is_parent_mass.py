import logging
from matchms import Spectrum
from matchms.filtering.require_parent_mass_match_smiles import _mass_diff_within_tolerance
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import derive_precursor_mz_from_parent_mass

logger = logging.getLogger("matchms")


def repair_precursor_is_parent_mass(spectrum_in: Spectrum,
                                    mass_tolerance):
    """Repairs parent mass and precursor mz if the parent mass is entered instead of the precursor_mz"""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Check if parent mass already matches smiles
    if _mass_diff_within_tolerance(spectrum.get("parent_mass"), spectrum.get("smiles"), mass_tolerance):
        return spectrum

    precursor_mz = spectrum.get("precursor_mz")
    smiles = spectrum.get("smiles")
    if _mass_diff_within_tolerance(precursor_mz, smiles, mass_tolerance):
        spectrum.set("parent_mass", precursor_mz)
        precursor_mz = derive_precursor_mz_from_parent_mass(spectrum)
        if precursor_mz is not None:
            spectrum.set("precursor_mz", precursor_mz)
    return spectrum
