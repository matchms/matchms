import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import \
    derive_precursor_mz_from_parent_mass
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from matchms.filtering.repair_parent_mass_from_smiles.require_parent_mass_match_smiles import \
    require_parent_mass_match_smiles


logger = logging.getLogger("matchms")


def repair_precursor_is_parent_mass(spectrum_in: Spectrum,
                                    mass_tolerance):
    """Repairs parent mass and precursor mz if the parent mass is entered instead of the precursor_mz"""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Check if parent mass already matches smiles
    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum

    precursor_mz = spectrum.get("precursor_mz")
    smiles = spectrum.get("smiles")
    smiles_mass = get_monoisotopic_neutral_mass(smiles)
    mass_difference = precursor_mz - smiles_mass
    if abs(mass_difference) < mass_tolerance:
        spectrum.set("parent_mass", precursor_mz)
        precursor_mz = derive_precursor_mz_from_parent_mass(spectrum)
        if precursor_mz is not None:
            spectrum.set("precursor_mz", precursor_mz)
    return spectrum
