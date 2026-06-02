import logging
from collections.abc import Mapping
from matchms.constants import PROTON_MASS
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_multiplier_and_mass_from_adduct,
)
from matchms.filtering.filter_utils.load_known_adducts import load_known_adducts
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)
from matchms.filtering.metadata_processing.clean_adduct import _clean_adduct
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def derive_parent_mass_from_metadata(
    metadata: Mapping,
    estimate_from_adduct: bool = True,
    estimate_from_charge: bool = True,
) -> float | None:
    """Use precursor m/z, charge, and adduct metadata to compute parent mass.
    
    Parameters
    ----------
    metadata
        Metadata dictionary containing at least precursor_mz and optionally charge and adduct.
    estimate_from_adduct
        Whether to attempt parent mass estimation based on adduct information.
    estimate_from_charge
        Whether to attempt parent mass estimation based on charge information.
    """
    if metadata is None:
        return None

    precursor_mz = as_float_or_none(metadata.get("precursor_mz"))
    if precursor_mz is None:
        logger.warning("Missing precursor m/z to derive parent mass.")
        return None

    charge = _get_charge_from_metadata(metadata)

    if estimate_from_adduct:
        multiplier, correction_mass = _get_multiplier_and_correction_mass_from_adduct(
            metadata.get("adduct")
        )
        if correction_mass is not None and multiplier is not None:
            return (precursor_mz - correction_mass) / multiplier

    if _is_valid_charge(charge) and estimate_from_charge:
        # Assume adduct of shape [M+xH] or [M-xH].
        protons_mass = PROTON_MASS * charge
        precursor_mass = precursor_mz * abs(charge)
        return precursor_mass - protons_mass

    return None


def _get_multiplier_and_correction_mass_from_adduct(adduct: str) -> tuple[int | None, float | None]:
    """Get mass multiplier and correction mass for an adduct."""
    adduct = as_string_or_none(adduct)
    if adduct is None:
        return None, None

    adduct = _clean_adduct(adduct)
    known_adducts = load_known_adducts()

    if adduct in list(known_adducts["adduct"]):
        matching_adduct = known_adducts[known_adducts["adduct"] == adduct]
        multiplier = matching_adduct["mass_multiplier"].values[0]
        correction_mass = matching_adduct["correction_mass"].values[0]
        return multiplier, correction_mass

    return get_multiplier_and_mass_from_adduct(adduct)


def _is_valid_charge(charge: int | float | None) -> bool:
    """Return True if a charge value can be used for parent-mass estimation."""
    return (charge is not None) and (charge != 0)


def _get_charge_from_metadata(metadata: Mapping):
    """Get charge from metadata.

    If no valid charge is found, guess +1 or -1 based on ionmode. Otherwise
    return 0.
    """
    charge = metadata.get("charge")

    if _is_valid_charge(charge):
        return charge

    ionmode = as_string_or_none(metadata.get("ionmode"))

    if ionmode == "positive":
        logger.info(
            "Missing charge entry, but positive ionmode detected. "
            "Consider prior run of `correct_charge()` filter."
        )
        return 1

    if ionmode == "negative":
        logger.info(
            "Missing charge entry, but negative ionmode detected. "
            "Consider prior run of `correct_charge()` filter."
        )
        return -1

    logger.warning(
        "Missing charge and ionmode entries. Consider prior run of "
        "`derive_ionmode()` and `correct_charge()` filters."
    )
    return 0