from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value


def _require_retention_index(metadata) -> bool:
    """Require retention index to be present.

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
        Spectrum input is returned unchanged if ``retention_index`` is present,
        otherwise ``None``. SpectraCollection input is returned with rows lacking
        retention index removed.
    """
    return not is_missing_metadata_value(metadata.get("retention_index"))


require_retention_index = metadata_requirement_filter(_require_retention_index)