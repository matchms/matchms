import logging
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_charge_of_adduct


logger = logging.getLogger("matchms")


def require_matching_adduct_and_ionmode(spectrum):
    if spectrum is None:
        return None
    ionmode = spectrum.get("ionmode")
    adduct = spectrum.get("adduct")
    charge_of_adduct = get_charge_of_adduct(adduct)
    if charge_of_adduct is None:
        return None
    if (charge_of_adduct > 0 and ionmode != "positive") or (charge_of_adduct < 0 and ionmode != "negative"):
        logger.warning("Ionmode: %s does not correspond to the charge or the adduct %s", ionmode, adduct)
        return None
    return spectrum
