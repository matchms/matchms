import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value


logger = logging.getLogger("matchms")


def _make_charge_int(metadata) -> dict:
    """Convert charge field to integer if possible.

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
        Input object with converted ``charge`` metadata, or ``None`` if the
        input was ``None``.
    """
    charge = metadata.get("charge", None)
    charge_int = _convert_charge_to_int(charge)

    if isinstance(charge_int, int):
        return {"charge": charge_int}

    return {}


def _convert_charge_to_int(charge):
    """Convert charge to integer if possible, else return None."""

    def _try_conversion(charge):
        try:
            return int(charge)
        except (ValueError, TypeError):
            logger.warning("Found charge (%s) cannot be converted to integer.", str(charge))
            return None

    if is_missing_metadata_value(charge):
        return None

    if isinstance(charge, int):
        return charge

    # Avoid pyteomics ChargeList and similar list-like charges.
    if isinstance(charge, list):
        if len(charge) == 0:
            return None
        return _try_conversion(charge[0])

    if isinstance(charge, str):
        charge = charge.strip().replace("+", "")
        if len(charge) > 1 and charge[-1] == "-":
            charge = "-" + charge.replace("-", "")
        return _try_conversion(charge)

    return _try_conversion(charge)


make_charge_int = metadata_update_filter(_make_charge_int)