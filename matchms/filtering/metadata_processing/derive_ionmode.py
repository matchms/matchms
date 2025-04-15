import logging
from typing import Optional
from matchms.Spectrum import Spectrum
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def derive_ionmode(spectrum_in: Spectrum, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Derive missing ionmode based on adduct.

    Some input formates (e.g. MGF files) do not always provide a correct ionmode.
    This function reads the adduct from the metadata and uses this to fill in the
    correct ionmode where missing.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum object with `ionmode` attribute set.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    ionmode = spectrum.get("ionmode")
    if ionmode in ["positive", "negative"]:
        return spectrum

    ionmode_from_charge = _derive_ionmode_from_charge(spectrum)
    ionmode_from_adduct = _derive_ionmode_from_adduct(spectrum)

    if ionmode_from_charge is not None and ionmode_from_adduct is not None:
        if ionmode_from_charge != ionmode_from_adduct:
            logger.warning(
                "The ionmode based on the charge (%s) does not match the ionmode based on the adduct (%s)",
                spectrum.get("charge"),
                spectrum.get("adduct"),
            )
            return spectrum
        spectrum.set("ionmode", ionmode_from_charge)
        logger.info("Set ionmode to %s based on the charge and adduct", ionmode_from_charge)
        return spectrum
    if ionmode_from_charge is not None and ionmode_from_adduct is None:
        spectrum.set("ionmode", ionmode_from_charge)
        logger.info("Set ionmode to %s based on the charge", ionmode_from_charge)
        return spectrum
    if ionmode_from_charge is None and ionmode_from_adduct is not None:
        spectrum.set("ionmode", ionmode_from_adduct)
        logger.info("Set ionmode to %s based on the charge", ionmode_from_adduct)
        return spectrum
    logger.info("The ionmode could not be derived from the charge or adduct")
    return spectrum


def _derive_ionmode_from_charge(spectrum):
    charge = spectrum.get("charge", None)

    if spectrum.get("charge") is None:
        return None
    if not isinstance(charge, int):
        logger.warning("Charge is given as string. Apply 'make_charge_int' filter first.")
        return None

    if charge > 0:
        return "positive"
    if charge < 0:
        return "negative"
    # In this case charge is 0
    return None


def _derive_ionmode_from_adduct(spectrum):
    adduct = spectrum.get("adduct", None)
    # Harmonize adduct string
    if adduct:
        adduct = _clean_adduct(adduct)

    # Load lists of known adducts
    known_adducts = load_known_adducts()
    # Try completing missing or incorrect ionmodes
    if adduct in list(known_adducts["adduct"]):
        ionmode = known_adducts.loc[known_adducts["adduct"] == adduct, "ionmode"].values[0]
        return ionmode
    return None
