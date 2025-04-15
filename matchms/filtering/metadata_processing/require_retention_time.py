import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_retention_time(
    spectrum_in: SpectrumType, minimum_rt=None, maximum_rt=None, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """
    This function checks if the input spectrum has a 'retention_time' in its metadata.
    If the input spectrum is None or doesn't have a 'retention_time', the function returns None.
    Otherwise, it returns a clone of the input spectrum.

    Parameters
    ----------
    spectrum_in (SpectrumType):
        The input spectrum to check.
    clone:
        Optionally clone the Spectrum.

    Returns:
    SpectrumType: A clone of the input spectrum if it has a 'retention_time', None otherwise.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    retention_time = spectrum.get("retention_time", None)
    if retention_time is None:
        return None
    if not isinstance(retention_time, (float, int)):
        logger.warning(
            "The retention time: %s is not a float or int, consider running add_retention first", str(retention_time)
        )
        return None
    if minimum_rt:
        if retention_time < minimum_rt:
            logger.info(
                "The retention time %s, was smaller than the minimum_rt %s and is therefore set to None",
                str(retention_time),
                str(minimum_rt),
            )
            return None
    if maximum_rt:
        if retention_time > maximum_rt:
            logger.info(
                "The retention time %s, was larger than the minimum_rt %s and is therefore set to None",
                str(retention_time),
                str(maximum_rt),
            )
            return None
    return spectrum
