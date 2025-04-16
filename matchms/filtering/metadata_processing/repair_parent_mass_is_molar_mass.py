import logging
from typing import Optional
from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_molecular_weight_neutral_mass,
    get_monoisotopic_neutral_mass,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def repair_parent_mass_is_molar_mass(
    spectrum_in: Spectrum, mass_tolerance: float, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Changes the parent mass from molar mass into monoistopic mass

    Manual entered parent mass is sometimes wrongly added as Molar mass instead of monoisotopic mass
    We check if the given parent mass is equal to the Molar mass (based on the smiles) and correct it to the
    monoisotopic mass in these cases.

    The molar mass is an average mass based on the average of all common isotopes and will therefore differ from what
    is measured in mass spectrometry.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum containing annotations to be checked and repaired.
    mass_tolerance:
        Maximum allowed mass difference between the calculated parent mass and the neutral
        monoisotopic mass derived from the SMILES.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with repaired parent mass, or `None` if not present.
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone() if clone else spectrum_in

    parent_mass = spectrum.get("parent_mass")
    smiles = spectrum.get("smiles")
    if smiles is None or parent_mass is None:
        return spectrum_in
    # Check if the parent mass is close to the molecular weight
    smiles_molecular_weight = get_molecular_weight_neutral_mass(smiles)
    if smiles_molecular_weight is None:
        return spectrum_in
    mass_difference = parent_mass - smiles_molecular_weight
    if abs(mass_difference) > mass_tolerance:
        # The parent mass is not close to the molecular weight
        return spectrum_in

    correct_mass = get_monoisotopic_neutral_mass(smiles)
    spectrum.set("parent_mass", correct_mass)
    logger.info(
        "Parent mass was molar mass instead of monoisotopic mass corrected from %s to %s", parent_mass, correct_mass
    )
    return spectrum
