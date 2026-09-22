import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import (
    get_monoisotopic_neutral_mass,
)
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from ..filter_utils.load_known_adducts import load_known_adducts
from .repair_adduct_based_on_parent_mass import _get_matching_adduct


logger = logging.getLogger("matchms")


def _repair_adduct_and_parent_mass_based_on_smiles(
    metadata,
    mass_tolerance: float,
) -> dict:
    """Correct adduct and parent mass based on smiles and precursor_mz."""
    smiles = as_string_or_none(metadata.get("smiles"))
    smiles_mass = get_monoisotopic_neutral_mass(smiles)

    if smiles_mass is None:
        return {}

    updates = {}
    parent_mass = as_float_or_none(metadata.get("parent_mass"))

    estimated_parent_mass = _estimate_parent_mass_from_adduct(metadata)

    need_to_update_adduct = False
    if estimated_parent_mass is not None:
        if abs(estimated_parent_mass - smiles_mass) > mass_tolerance:
            need_to_update_adduct = True
    else:
        need_to_update_adduct = True

    if need_to_update_adduct:
        new_adduct = _get_matching_adduct(
            precursor_mz=metadata.get("precursor_mz"),
            parent_mass=smiles_mass,
            ion_mode=metadata.get("ionmode"),
            mass_tolerance=mass_tolerance,
        )

        # Preserve old behavior: if the adduct needs fixing but no matching
        # adduct is found, do not update parent_mass either.
        if new_adduct is None:
            return {}

        current_adduct = metadata.get("adduct")
        if new_adduct != current_adduct:
            updates["adduct"] = new_adduct
            logger.info("Adduct was set from %s to %s", current_adduct, new_adduct)

    if parent_mass is None:
        updates["parent_mass"] = smiles_mass
        logger.info("Parent mass was set to match the smiles mass: %s", smiles_mass)
    elif abs(smiles_mass - parent_mass) > mass_tolerance:
        updates["parent_mass"] = smiles_mass
        logger.info(
            "Parent mass was updated from %s to %s to match the smiles mass",
            parent_mass,
            smiles_mass,
        )

    return updates


def _estimate_parent_mass_from_adduct(metadata):
    """Estimate parent mass from precursor_mz and adduct metadata."""
    precursor_mz = as_float_or_none(metadata.get("precursor_mz"))
    adduct = as_string_or_none(metadata.get("adduct"))

    if precursor_mz is None or adduct is None:
        return None

    adducts_df = load_known_adducts()
    match = adducts_df[adducts_df["adduct"] == adduct]

    if match.empty:
        return None

    adduct_row = match.iloc[0]
    return (
        precursor_mz - adduct_row["correction_mass"]
    ) / adduct_row["mass_multiplier"]


repair_adduct_and_parent_mass_based_on_smiles = metadata_update_filter(
    _repair_adduct_and_parent_mass_based_on_smiles
)