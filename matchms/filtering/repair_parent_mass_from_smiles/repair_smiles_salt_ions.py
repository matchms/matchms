import itertools
import logging
from matchms.filtering.repair_parent_mass_from_smiles.repair_precursor_is_parent_mass import _get_monoisotopic_neutral_mass
from matchms.filtering.repair_parent_mass_from_smiles.repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt

logger = logging.getLogger("matchms")


def repair_smiles_salt_ions(spectrum_in,
                            mass_tolerance,
                            accept_parent_mass_is_mol_wt = False):
    """Checks if parent mass matches one of the ions"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    smiles = spectrum.get("smiles")
    parent_mass = spectrum.get("parent_mass")
    possible_ion_combinations = create_possible_ions(smiles)

    for ion, not_used_ions in possible_ion_combinations:
        spectrum_with_ions = spectrum.clone()
        spectrum_with_ions.set("smiles", ion)
        spectrum_with_ions.set("salt_ions", not_used_ions)
        if accept_parent_mass_is_mol_wt:
            spectrum_with_ions = repair_parent_mass_is_mol_wt(spectrum_with_ions, mass_tolerance)
        smiles_mass = _get_monoisotopic_neutral_mass(smiles)
        if abs(parent_mass - smiles_mass) < mass_tolerance:
            logger.info(f"Removed salt ions: {not_used_ions} from {smiles} to match parent mass")
            return spectrum_with_ions
    return spectrum


def create_possible_ions(smiles):
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
                results.append((combined_ion, removed_ions))
    return results
