import logging
import re
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _derive_formula_from_name(
    metadata,
    remove_formula_from_name: bool = True,
) -> dict:
    """Detect and remove misplaced formula in compound name and add to metadata.

    Method to find misplaced formulas in compound name based on regular
    expression. This will not chemically test the detected formula, so the
    search is limited to frequently occurring types of shape ``C47H83N1O8P1``.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    remove_formula_from_name
        Remove found formula from compound name if set to ``True``.
        Default is ``True``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with added ``formula`` metadata and optionally cleaned
        ``compound_name``, or ``None`` if the input was ``None``.
    """
    name = as_string_or_none(metadata.get("compound_name"))

    if name is None:
        fallback_name = as_string_or_none(metadata.get("name"))
        assert fallback_name in [None, ""], (
            "Found 'name' but not 'compound_name' in metadata",
            "Apply 'add_compound_name' filter first.",
        )
        return {}

    end_of_name = name.split(" ")[-1]
    formula_from_name = end_of_name if _looks_like_formula(end_of_name) else None

    if formula_from_name is None:
        return {}

    updates = {}

    if remove_formula_from_name:
        name_formula_removed = " ".join(name.split(" ")[:-1])
        updates["compound_name"] = name_formula_removed
        logger.info("Removed formula %s from compound name.", formula_from_name)

    if metadata.get("formula", None) is None:
        updates["formula"] = formula_from_name
        logger.info("Added formula %s to metadata.", formula_from_name)

    return updates


def _looks_like_formula(formula):
    """Return True if input string has expected molecular formula format."""
    if not isinstance(formula, str):
        return False

    regex_atoms = r"([CFHNOPS])"
    atom_count = len(re.findall(regex_atoms, formula))
    regexp = r"^([CFHNOPS]|[0-9]|\(|\)){3,}$"

    return (atom_count > 2) and (re.search(regexp, formula) is not None)


derive_formula_from_name = metadata_update_filter(_derive_formula_from_name)