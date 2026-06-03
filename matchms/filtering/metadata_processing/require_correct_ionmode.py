import logging
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _require_correct_ionmode(metadata, ion_mode_to_keep) -> bool:
    """Validate that the spectrum ionmode matches the requested ionmode.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    ion_mode_to_keep
        Desired ionmode: ``"positive"``, ``"negative"``, or ``"both"``.
        If ``"both"``, spectra are kept when ionmode is either ``"positive"``
        or ``"negative"``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if its ionmode matches the
        requirement, otherwise ``None``. SpectraCollection input is returned with
        non-matching rows removed.
    """
    if ion_mode_to_keep not in {"positive", "negative", "both"}:
        raise ValueError("ion_mode_to_keep should be 'positive', 'negative' or 'both'")

    ionmode = as_string_or_none(metadata.get("ionmode"))

    if ion_mode_to_keep == "both":
        if ionmode in ("positive", "negative"):
            return True

        logger.info(
            "Spectrum was removed since ionmode was: %s which does not match positive or negative",
            ionmode,
        )
        return False

    if ionmode == ion_mode_to_keep:
        return True

    logger.info(
        "Spectrum was removed since ionmode was: %s which does not match %s",
        ionmode,
        ion_mode_to_keep,
    )
    return False


require_correct_ionmode = metadata_requirement_filter(_require_correct_ionmode)