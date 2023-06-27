import logging
from matchms import Spectrum


logger = logging.getLogger("matchms")


def require_correct_ionmode(spectrum_in: Spectrum,
                            ion_mode_to_keep):
    """Removes spectra that are not in the correct ionmode"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    assert ion_mode_to_keep in {"positive", "negative", "both"}, \
        "ion_mode should be set to 'positive', 'negative' or 'both'"
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
