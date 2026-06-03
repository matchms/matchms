from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)


def _repair_parent_mass_from_smiles(
    metadata,
    mass_tolerance: float = 0.1,
) -> dict:
    """Set parent mass to match smiles mass if not already close."""
    smiles = as_string_or_none(metadata.get("smiles"))
    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None:
        return {}

    parent_mass = as_float_or_none(metadata.get("parent_mass"))

    if parent_mass is None:
        return {"parent_mass": smiles_mass}

    if abs(parent_mass - smiles_mass) > mass_tolerance:
        return {"parent_mass": smiles_mass}

    return {}


repair_parent_mass_from_smiles = metadata_update_filter(_repair_parent_mass_from_smiles)