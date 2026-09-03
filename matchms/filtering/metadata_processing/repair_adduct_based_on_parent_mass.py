import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from ..filter_utils.load_known_adducts import load_known_adducts


logger = logging.getLogger("matchms")


def _repair_adduct_based_on_parent_mass(metadata, mass_tolerance: float) -> dict:
    """Correct adduct based on parent_mass and precursor_mz."""
    current_adduct = metadata.get("adduct")

    new_adduct = _get_matching_adduct(
        precursor_mz=metadata.get("precursor_mz"),
        parent_mass=metadata.get("parent_mass"),
        ion_mode=metadata.get("ionmode"),
        mass_tolerance=mass_tolerance,
    )

    if new_adduct is None:
        return {}

    if new_adduct != current_adduct:
        logger.info("Adduct was set from %s to %s", current_adduct, new_adduct)
        return {"adduct": new_adduct}

    return {}


def _get_matching_adduct(precursor_mz, parent_mass, ion_mode, mass_tolerance):
    precursor_mz = as_float_or_none(precursor_mz)
    parent_mass = as_float_or_none(parent_mass)
    ion_mode = as_string_or_none(ion_mode)

    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return None

    if ion_mode not in ("positive", "negative"):
        if ion_mode is not None:
            logger.warning(
                "Ionmode: %s not positive, negative or None, first run derive_ionmode",
                ion_mode,
            )
        return None

    if parent_mass is None:
        return None

    adducts_df = load_known_adducts()
    adducts_df = adducts_df[adducts_df["ionmode"] == ion_mode]

    # Do not use M+ and M- because these could accidentally repair cases where
    # parent_mass was filled into precursor_mz.
    adducts_df = adducts_df[~adducts_df["adduct"].isin(("[M]+", "[M]-"))]

    parent_masses = (
        precursor_mz - adducts_df["correction_mass"]
    ) / adducts_df["mass_multiplier"]
    mass_differences = abs(parent_masses - parent_mass)

    smallest_mass_index = mass_differences.idxmin()
    adduct = adducts_df.loc[smallest_mass_index]["adduct"]

    if mass_differences[smallest_mass_index] < mass_tolerance:
        return adduct

    return None


repair_adduct_based_on_parent_mass = metadata_update_filter(
    _repair_adduct_based_on_parent_mass
)