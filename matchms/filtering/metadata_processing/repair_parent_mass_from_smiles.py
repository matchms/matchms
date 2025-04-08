import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import get_monoisotopic_neutral_mass


logger = logging.getLogger("matchms")


def repair_parent_mass_from_smiles(spectrum_in: Spectrum, mass_tolerance: float = 0.1):
    """Sets the parent mass to match the smiles mass, if not already close to smiles mass"""
    if spectrum_in is None:
        return None
    changed_spectrum = spectrum_in.clone()
    smiles_mass = get_monoisotopic_neutral_mass(changed_spectrum.get("smiles"))
    if smiles_mass is None:
        return spectrum_in
    parent_mass = spectrum_in.get("parent_mass")

    if parent_mass is None:
        changed_spectrum.set("parent_mass", smiles_mass)
        return changed_spectrum
    if abs(parent_mass - smiles_mass) > mass_tolerance:
        changed_spectrum.set("parent_mass", smiles_mass)
        return changed_spectrum
    return spectrum_in
