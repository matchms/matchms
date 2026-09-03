import logging
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none
from ..filter_utils.load_known_adducts import load_known_adducts
from .clean_adduct import _clean_adduct


logger = logging.getLogger("matchms")


def _derive_ionmode(metadata) -> dict:
    """Derive missing ionmode based on charge and/or adduct.

    Some input formats, for example MGF files, do not always provide a correct
    ionmode. This filter reads charge and adduct metadata and uses them to fill
    in the ionmode where missing.
    """
    ionmode = as_string_or_none(metadata.get("ionmode"))

    if ionmode in ("positive", "negative"):
        return {}

    ionmode_from_charge = _derive_ionmode_from_charge(metadata.get("charge"))
    ionmode_from_adduct = _derive_ionmode_from_adduct(metadata.get("adduct"))

    if ionmode_from_charge is not None and ionmode_from_adduct is not None:
        if ionmode_from_charge != ionmode_from_adduct:
            logger.warning(
                "The ionmode based on the charge (%s) does not match the ionmode based on the adduct (%s)",
                metadata.get("charge"),
                metadata.get("adduct"),
            )
            return {}

        logger.info("Set ionmode to %s based on the charge and adduct", ionmode_from_charge)
        return {"ionmode": ionmode_from_charge}

    if ionmode_from_charge is not None:
        logger.info("Set ionmode to %s based on the charge", ionmode_from_charge)
        return {"ionmode": ionmode_from_charge}

    if ionmode_from_adduct is not None:
        logger.info("Set ionmode to %s based on the adduct", ionmode_from_adduct)
        return {"ionmode": ionmode_from_adduct}

    logger.info("The ionmode could not be derived from the charge or adduct")
    return {}


def _derive_ionmode_from_charge(charge):
    if charge is None:
        return None

    if not isinstance(charge, int):
        logger.warning("Charge is given as string. Apply 'make_charge_int' filter first.")
        return None

    if charge > 0:
        return "positive"
    if charge < 0:
        return "negative"

    return None


def _derive_ionmode_from_adduct(adduct):
    adduct = as_string_or_none(adduct)

    if adduct:
        adduct = _clean_adduct(adduct)

    known_adducts = load_known_adducts()

    if adduct in list(known_adducts["adduct"]):
        return known_adducts.loc[
            known_adducts["adduct"] == adduct,
            "ionmode",
        ].values[0]

    return None


derive_ionmode = metadata_update_filter(_derive_ionmode)