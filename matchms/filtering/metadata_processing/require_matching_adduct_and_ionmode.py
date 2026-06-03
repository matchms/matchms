import logging
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_charge_of_adduct
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _require_matching_adduct_and_ionmode(metadata) -> bool:
    """Remove spectra where the adduct and ionmode do not match.

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
        Spectrum input is returned unchanged if adduct and ionmode match,
        otherwise ``None``. SpectraCollection input is returned with non-matching
        rows removed.
    """
    ionmode = as_string_or_none(metadata.get("ionmode"))
    adduct = as_string_or_none(metadata.get("adduct"))

    charge_of_adduct = get_charge_of_adduct(adduct)
    if charge_of_adduct is None:
        return False

    if (charge_of_adduct > 0 and ionmode != "positive") or (
        charge_of_adduct < 0 and ionmode != "negative"
    ):
        logger.warning(
            "Ionmode: %s does not correspond to the charge or the adduct %s",
            ionmode,
            adduct,
        )
        return False

    return True


require_matching_adduct_and_ionmode = metadata_requirement_filter(
    _require_matching_adduct_and_ionmode
)