import itertools
import logging
from typing import Optional
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import get_monoisotopic_neutral_mass
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import is_valid_smiles
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def repair_smiles_of_salts(spectrum_in, mass_tolerance, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Repairs the smiles of a salt to match the parent mass.
    E.g. C1=NC2=NC=NC(=C2N1)N.Cl is converted to 1=NC2=NC=NC(=C2N1)N if this matches the parent mass
    Checks if parent mass matches one of the ions

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    mass_tolerance:
        Maximum allowed mass difference between the calculated parent mass and the neutral
        monoisotopic mass derived from the SMILES.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with repaired SMILES, or `None` if not present.
    """
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone() if clone else spectrum_in

    smiles = spectrum.get("smiles")
    if smiles is None:
        return spectrum

    if not is_valid_smiles(smiles):
        return spectrum
    parent_mass = spectrum.get("parent_mass")
    possible_ion_combinations = _create_possible_ions(smiles)
    if not possible_ion_combinations:
        # It is not a salt
        return spectrum
    for ion, not_used_ions in possible_ion_combinations:
        ion_mass = get_monoisotopic_neutral_mass(ion)
        if ion_mass is None:
            continue
        mass_diff = abs(parent_mass - ion_mass)
        # Check for Repair parent mass is mol wt did only return 1 spectrum. So not added as option for simplicity.
        if mass_diff < mass_tolerance:
            spectrum_with_ions = spectrum.clone()
            spectrum_with_ions.set("smiles", ion)
            spectrum_with_ions.set("salt_ions", not_used_ions)
            logger.info("Removed salt ions: %s from %s to match parent mass", not_used_ions, smiles)
            return spectrum_with_ions
    logger.warning("None of the parts of the smile %s match the parent mass: %s", smiles, parent_mass)
    return spectrum


def _create_possible_ions(smiles):
    """Selects all possible ion combinations of a salt"""

    results = []
    if "." in smiles:
        single_ions = smiles.split(".")
        for r in range(1, len(single_ions) + 1):
            combinations = itertools.combinations(single_ions, r)
            for combination in combinations:
                combined_ion = ".".join(combination)
                removed_ions = single_ions.copy()
                for used_ion in combination:
                    removed_ions.remove(used_ion)
                removed_ions = ".".join(removed_ions)
                results.append((combined_ion, removed_ions))
    return results
