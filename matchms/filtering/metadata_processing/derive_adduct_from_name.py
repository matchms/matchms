import logging
import re
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.interpret_unknown_adduct import (
    get_multiplier_and_mass_from_adduct,
)
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def _derive_adduct_from_name(
    metadata,
    remove_adduct_from_name: bool = True,
) -> dict:
    """Find adduct in compound name and add it to metadata if not present yet.

    Method to interpret the given compound name to find the adduct.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    remove_adduct_from_name
        Remove found adducts from compound name if set to ``True``.
        Default is ``True``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with added ``adduct`` metadata, or ``None`` if the input
        was ``None``.
    """
    compound_name = as_string_or_none(metadata.get("compound_name"))

    if compound_name is None:
        if as_string_or_none(metadata.get("name")) not in [None, ""]:
            logger.warning(
                "Found 'name' but not 'compound_name' in metadata. "
                "Apply 'add_compound_name' filter first."
            )
        return {}

    parts_that_look_like_adduct = []
    name_split = compound_name.split(" ")

    for name_part in name_split:
        if _looks_like_adduct(name_part):
            # Some adducts occur more than once. So they are all removed.
            parts_that_look_like_adduct.append(name_part)

    if len(parts_that_look_like_adduct) == 0:
        return {}

    updates = {}

    if remove_adduct_from_name:
        name_adduct_removed = " ".join(
            x for x in name_split if x not in parts_that_look_like_adduct
        )
        name_adduct_removed = name_adduct_removed.strip("; ")
        updates["compound_name"] = name_adduct_removed
        logger.info(
            "Removed adduct %s from compound name.",
            parts_that_look_like_adduct,
        )

    if not _looks_like_adduct(metadata.get("adduct")):
        best_adduct = _select_best_adduct(parts_that_look_like_adduct)
        if best_adduct:
            updates["adduct"] = best_adduct
            logger.info(
                "Added adduct %s from the compound name to metadata.",
                best_adduct,
            )

    return updates


def _select_best_adduct(list_of_adducts: list[str]) -> str | None:
    """Select an adduct that can actually be interpreted."""
    unique_cleaned_adducts = list({_clean_adduct(adduct) for adduct in list_of_adducts})

    if len(unique_cleaned_adducts) == 1:
        return unique_cleaned_adducts[0]

    completely_correct_adduct = []
    for adduct in unique_cleaned_adducts:
        multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
        if multiplier and correction_mass:
            completely_correct_adduct.append(adduct)

    if len(completely_correct_adduct) == 0:
        return None

    if len(completely_correct_adduct) == 1:
        return completely_correct_adduct[0]

    logger.warning(
        "Two potential adducts were found in the compound name that are both "
        "valid adducts. The first adduct is used. The adducts found are: %s",
        completely_correct_adduct,
    )
    return completely_correct_adduct[0]


def _looks_like_adduct(adduct):
    """Return True if input string has expected format of an adduct."""
    if not isinstance(adduct, str):
        return False

    adduct = _clean_adduct(adduct)

    known_adducts = load_known_adducts()
    if adduct in list(known_adducts["adduct"]):
        return True

    regexp1 = (
        r"^\[(([0-4]M)|(M[0-9])|(M))"
        r"((Br)|(Br81)|(Cl)|(Cl37)|(S)){0,}"
        r"[+-][A-Z0-9\+\-\(\)aglire]{1,}[\]0-4+-]{1,4}"
    )
    return re.search(regexp1, adduct) is not None


derive_adduct_from_name = metadata_update_filter(_derive_adduct_from_name)