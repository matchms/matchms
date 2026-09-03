import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_molecular_weight_neutral_mass,
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)


logger = logging.getLogger("matchms")


def _repair_parent_mass_is_molar_mass(metadata, mass_tolerance: float) -> dict:
    """Change parent mass from molar mass into monoisotopic mass where applicable."""
    smiles = as_string_or_none(metadata.get("smiles"))
    parent_mass = as_float_or_none(metadata.get("parent_mass"))

    if smiles is None or parent_mass is None:
        return {}

    smiles_molecular_weight = get_molecular_weight_neutral_mass(smiles)
    if smiles_molecular_weight is None:
        return {}

    if abs(parent_mass - smiles_molecular_weight) > mass_tolerance:
        return {}

    correct_mass = get_monoisotopic_neutral_mass(smiles)
    if correct_mass is None:
        return {}

    logger.info(
        "Parent mass was molar mass instead of monoisotopic mass corrected from %s to %s",
        parent_mass,
        correct_mass,
    )

    return {"parent_mass": correct_mass}


repair_parent_mass_is_molar_mass = metadata_update_filter(
    _repair_parent_mass_is_molar_mass
)