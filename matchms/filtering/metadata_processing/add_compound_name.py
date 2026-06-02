import logging
from matchms.filtering._dispatch import metadata_update_filter


logger = logging.getLogger("matchms")


def _add_compound_name(metadata) -> dict:
    """Add compound name to the ``compound_name`` metadata field.

    If ``compound_name`` is missing, this filter tries to copy the value from
    ``name`` first and then from ``title``.

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
        Input object with added ``compound_name`` metadata, or ``None`` if the
        input was ``None``.
    """
    if metadata.get("compound_name") is not None:
        return {}

    if isinstance(metadata.get("name"), str):
        return {"compound_name": metadata.get("name")}

    if isinstance(metadata.get("title"), str):
        return {"compound_name": metadata.get("title")}

    logger.info("No compound name found in metadata.")
    return {}


add_compound_name = metadata_update_filter(_add_compound_name)