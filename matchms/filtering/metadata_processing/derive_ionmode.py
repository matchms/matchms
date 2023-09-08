import logging
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def derive_ionmode(spectrum_in: SpectrumType) -> SpectrumType:
    """Derive missing ionmode based on adduct.

    Some input formates (e.g. MGF files) do not always provide a correct ionmode.
    This function reads the adduct from the metadata and uses this to fill in the
    correct ionmode where missing.

    Parameters
    ----------
    spectrum
        Input spectrum.

    Returns
    -------
    Spectrum object with `ionmode` attribute set.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    ionmode = spectrum.get("ionmode")
    if ionmode in ["positive", "negative"]:
        return spectrum

    adduct = spectrum.get("adduct", None)
    # Harmonize adduct string
    if adduct:
        adduct = _clean_adduct(adduct)

    # Load lists of known adducts
    known_adducts = load_known_adducts()
    # Try completing missing or incorrect ionmodes
    if adduct in list(known_adducts["adduct"]):
        ionmode = known_adducts.loc[known_adducts["adduct"] == adduct, "ionmode"].values[0]
    else:
        ionmode = None

    spectrum.set("ionmode", ionmode)
    logger.info("Set ionmode to %s.", ionmode)
    return spectrum
