import logging
import numpy as np
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    as_string_or_none,
)


logger = logging.getLogger("matchms")


def _correct_charge(metadata) -> dict:
    """Correct charge values based on given ionmode.

    For some spectra, the charge value is either undefined or inconsistent with
    its ionmode, which is corrected by this filter.

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
        Input object with corrected ``charge`` metadata, or ``None`` if the
        input was ``None``.
    """
    raw_ionmode = metadata.get("ionmode", None)
    ionmode = as_string_or_none(raw_ionmode)

    if ionmode is not None and ionmode != ionmode.lower():
        raise ValueError(
            "Ionmode field not harmonized. "
            "Apply 'make_ionmode_lowercase' filter first."
        )

    raw_charge = metadata.get("charge", None)
    if isinstance(raw_charge, str):
        raise ValueError(
            "Charge is given as string. Apply 'make_charge_int' filter first."
        )

    charge = as_float_or_none(raw_charge)
    if charge is None:
        charge = 0

    if charge == 0 and ionmode == "positive":
        charge = 1
        logger.info("Guessed charge to 1 based on positive ionmode")
    elif charge == 0 and ionmode == "negative":
        charge = -1
        logger.info("Guessed charge to -1 based on negative ionmode")

    if np.sign(charge) == 1 and ionmode == "negative":
        logger.info(
            "Changed sign of given charge: %s to match negative ionmode",
            charge,
        )
        charge *= -1
    elif np.sign(charge) == -1 and ionmode == "positive":
        logger.warning(
            "Changed sign of given charge: %s to match positive ionmode",
            charge,
        )
        charge *= -1

    if float(charge).is_integer():
        charge = int(charge)

    return {"charge": charge}


correct_charge = metadata_update_filter(_correct_charge)