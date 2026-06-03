import logging
import re
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _require_formula(metadata) -> bool:
    """Ensure that the molecular formula is present and looks valid.

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
        Spectrum input is returned unchanged if it contains a valid molecular
        formula, otherwise ``None``. SpectraCollection input is returned with
        rows lacking a valid formula removed.
    """
    formula = as_string_or_none(metadata.get("formula"))

    if formula and _is_valid_formula(formula):
        return True

    logger.info("Spectrum does not contain a valid molecular formula.")
    return False


def _is_valid_formula(formula: str) -> bool:
    """Check if string looks like a valid chemical formula."""
    pattern = r"^([A-Z][a-z]?\d*)+$"
    return re.fullmatch(pattern, formula) is not None


require_formula = metadata_requirement_filter(_require_formula)