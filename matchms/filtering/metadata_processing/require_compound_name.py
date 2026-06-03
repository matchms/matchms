import logging
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _require_compound_name(metadata) -> bool:
    """Ensure that the compound name is present in the spectrum metadata.

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
        Spectrum input is returned unchanged if it contains a compound name,
        otherwise ``None``. SpectraCollection input is returned with rows lacking
        compound names removed.
    """
    compound_name = as_string_or_none(metadata.get("compound_name"))

    if compound_name:
        return True

    logger.info("Spectrum does not contain a compound name.")
    return False


require_compound_name = metadata_requirement_filter(_require_compound_name)