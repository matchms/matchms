import itertools
import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_smiles,
)


logger = logging.getLogger("matchms")


def _repair_smiles_of_salts(metadata, mass_tolerance: float) -> dict:
    """Repair salt SMILES to match parent mass."""
    smiles = metadata.get("smiles")
    parent_mass = metadata.get("parent_mass")

    ion, not_used_ions = _find_matching_salt_ion(
        smiles=smiles,
        parent_mass=parent_mass,
        mass_tolerance=mass_tolerance,
    )

    if ion is None:
        return {}

    logger.info(
        "Removed salt ions: %s from %s to match parent mass",
        not_used_ions,
        smiles,
    )

    return {
        "smiles": ion,
        "salt_ions": not_used_ions,
    }


def _find_matching_salt_ion(smiles, parent_mass, mass_tolerance):
    """Return matching ion and removed ions if one salt part matches parent mass."""
    smiles = as_string_or_none(smiles)
    parent_mass = as_float_or_none(parent_mass)

    if smiles is None or parent_mass is None:
        return None, None

    if not is_valid_smiles(smiles):
        return None, None

    possible_ion_combinations = _create_possible_ions(smiles)
    if not possible_ion_combinations:
        return None, None

    for ion, not_used_ions in possible_ion_combinations:
        ion_mass = get_monoisotopic_neutral_mass(ion)
        if ion_mass is None:
            continue

        mass_diff = abs(parent_mass - ion_mass)
        if mass_diff < mass_tolerance:
            return ion, not_used_ions

    logger.warning(
        "None of the parts of the smile %s match the parent mass: %s",
        smiles,
        parent_mass,
    )
    return None, None


def _create_possible_ions(smiles):
    """Select all possible ion combinations of a salt."""
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


repair_smiles_of_salts = metadata_update_filter(_repair_smiles_of_salts)