from typing import Optional
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import get_monoisotopic_neutral_mass
from matchms.typing import SpectrumType


def require_parent_mass_match_smiles(spectrum_in: SpectrumType, mass_tolerance) -> Optional[SpectrumType]:
    """
    Validates if the parent mass of the given spectrum matches the mass calculated
    from its associated SMILES string within a specified tolerance.

    Parameters
    ----------
    spectrum_in: Spectrum
        The input spectrum to be validated. If `None`, the function will return `None`.

    mass_tolerance: float
        The tolerance for the mass difference between the spectrum's parent mass and
        the mass calculated from its SMILES string.

    Returns
    -------
    Spectrum or None
        The validated spectrum if its parent mass matches the SMILES mass within the
        specified tolerance, or `None` otherwise.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in

    # Check if parent mass matches the smiles mass
    if _check_smiles_and_parent_mass_match(spectrum.get("smiles"), spectrum.get("parent_mass"), mass_tolerance):
        return spectrum
    return None


def _check_smiles_and_parent_mass_match(smiles, parent_mass, mass_tolerance) -> bool:
    """Returns True if smiles and parent mass are matching"""
    smiles_mass = get_monoisotopic_neutral_mass(smiles)
    if smiles_mass is None or parent_mass is None:
        return False
    mass_difference = parent_mass - smiles_mass
    if abs(mass_difference) < mass_tolerance:
        return True
    return False
