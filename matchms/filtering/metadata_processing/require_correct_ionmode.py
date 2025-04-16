import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_correct_ionmode(spectrum_in: SpectrumType, ion_mode_to_keep) -> Optional[SpectrumType]:
    """
    Validates the ion mode of a given spectrum. If the spectrum's ion mode
    doesn't match the `ion_mode_to_keep`, it will be removed and a log message
    will be generated.

    Parameters
    ----------
    spectrum_in: Spectrum
        The input spectrum to be validated. If `None`, the function will return `None`.

    ion_mode_to_keep: str
        Desired ion mode ('positive', 'negative', or 'both'). If not one of these,
        a `ValueError` is raised.

    Returns
    -------
    Spectrum or None
        The validated spectrum if its ion mode matches the desired one, or `None` otherwise.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in

    if ion_mode_to_keep not in {"positive", "negative", "both"}:
        raise ValueError("ion_mode_to_keep should be 'positive', 'negative' or 'both'")
    ion_mode = spectrum.get("ionmode")
    if ion_mode_to_keep == "both":
        if ion_mode in ("positive", "negative"):
            return spectrum

        logger.info("Spectrum was removed since ionmode was: %s which does not match positive or negative", ion_mode)
        return None
    if ion_mode == ion_mode_to_keep:
        return spectrum
    logger.info("Spectrum was removed since ionmode was: %s which does not match %s", ion_mode, ion_mode_to_keep)
    return None
