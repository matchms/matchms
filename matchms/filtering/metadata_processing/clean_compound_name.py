import logging
import re
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _clean_compound_name(metadata) -> dict:
    """Clean compound name.

    A list of frequently seen name additions that do not belong to the compound
    name will be removed.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with cleaned ``compound_name`` metadata, or ``None`` if the
        input was ``None``.
    """
    name = as_string_or_none(metadata.get("compound_name"))

    if name is None:
        fallback_name = as_string_or_none(metadata.get("name"))
        assert fallback_name in [None, ""], (
            "Found 'name' but not 'compound_name' in metadata",
            "Apply 'add_compound_name' filter first.",
        )
        return {}

    name_cleaned = _remove_parts_by_regular_expression(name)
    name_cleaned = _remove_known_non_compound_parts(name_cleaned)
    name_cleaned = _remove_misplaced_mass(name_cleaned)

    if name_cleaned != name:
        logger.info("Added cleaned compound name: %s", name_cleaned)
        return {"compound_name": name_cleaned}

    return {}


def _remove_parts_by_regular_expression(name: str):
    """Clean name string by removing known parts that do not belong there."""
    name = name.strip()

    name = re.split(r"[A-Z]{3,6}[0-9]{8,12}-[0-9]{2,5}_[A-Z,0-9]{4,15}_", name)[-1]
    name = re.split(r"[A-Z]{3,6}[0-9]{8,12}-[0-9]{2,3}\!", name)[-1]
    name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}.*\|", name)[-1]
    name = re.split(r"((Massbank:)|(MassbankEU:))[A-Z]{2}[0-9]{5,6}", name)[-1]
    name = re.split(r"HMDB:HMDB[0-9]{4,7}-[0-9]{1,7}", name)[-1]
    name = re.split(r"MoNA:[0-9]{5,10}", name)[-1]
    name = re.split(r"ReSpect:[A-Z]{2,3}[0-9]{6}.*\|", name)[-1]
    name = re.split(r"[A-Z]{2,3}[0-9]{6}( )", name)[-1]
    name = re.split(r"^[0-9]{4}_", name)[-1]
    name = re.split(r"_((HCD)|(CID))[0-9]{2}_[0-9]{5,6}$", name)[0]
    name = re.split(r"(?: - )?[0-9]+(?:\.[0-9]+)? ?[eE][Vv](?: Unknown)?$", name)[0]

    return name


def _remove_known_non_compound_parts(name: str):
    """Remove known non compound-name strings from name."""
    parts_remove = ["Spectral Match to", "from NIST14", "Massbank:"]
    for part in parts_remove:
        name = name.replace(part, "")
    return name.strip("; ")


def _remove_misplaced_mass(name: str):
    """Remove occasionally occurring parent mass addition to name."""
    regex_mass = r"^[0-9]{2,4}\.[0-9]$"
    end_part = name.split(" ")[-1]

    if re.search(regex_mass, end_part) is not None:
        return name.replace(end_part, "").strip()

    return name


clean_compound_name = metadata_update_filter(_clean_compound_name)