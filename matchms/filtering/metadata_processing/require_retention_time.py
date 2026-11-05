import logging
import numbers
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import is_missing_metadata_value


logger = logging.getLogger("matchms")


def _require_retention_time(metadata, minimum_rt=None, maximum_rt=None) -> bool:
    """Require retention time to be present and within optional bounds.

    Parameters
    ----------
    spectrum_in
        Input :class:`~matchms.spectrum.Spectrum` or :class:`~matchms.spectra_collection.SpectraCollection`.
    minimum_rt
        Minimum accepted retention time. Default is ``None`` (no lower bound).
    maximum_rt
        Maximum accepted retention time. Default is ``None`` (no upper bound).
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if ``retention_time`` is present
        and within the bounds, otherwise ``None``. SpectraCollection input is
        returned with rows lacking a valid retention time removed.
    """
    retention_time = metadata.get("retention_time", None)

    if is_missing_metadata_value(retention_time):
        return False

    if not isinstance(retention_time, numbers.Real):
        logger.warning(
            "The retention time: %s is not a float or int, consider running add_retention first",
            str(retention_time),
        )
        return False

    if minimum_rt is not None and retention_time < minimum_rt:
        logger.info(
            "The retention time %s, was smaller than the minimum_rt %s and is therefore set to None",
            str(retention_time),
            str(minimum_rt),
        )
        return False

    if maximum_rt is not None and retention_time > maximum_rt:
        logger.info(
            "The retention time %s, was larger than the minimum_rt %s and is therefore set to None",
            str(retention_time),
            str(maximum_rt),
        )
        return False

    return True


require_retention_time = metadata_requirement_filter(_require_retention_time)
