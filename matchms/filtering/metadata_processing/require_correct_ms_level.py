from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


def _require_correct_ms_level(metadata, required_ms_level: int = 2) -> bool:
    """Remove spectra where the ms_level does not match the required_ms_level.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    required_ms_level
        Required MS level. Default is ``2``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if the MS level matches, otherwise
        ``None``. SpectraCollection input is returned with non-matching rows
        removed.
    """
    ms_level = as_string_or_none(metadata.get("ms_level"))

    return ms_level in (f"MS{required_ms_level}", str(required_ms_level))


require_correct_ms_level = metadata_requirement_filter(_require_correct_ms_level)