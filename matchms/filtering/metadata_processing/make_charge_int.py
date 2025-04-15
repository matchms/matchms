import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def make_charge_int(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Convert charge field to integer (if possible).

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with converted charge, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    charge = spectrum.get("charge", None)
    charge_int = _convert_charge_to_int(charge)
    if isinstance(charge_int, int):
        spectrum.set("charge", charge_int)

    return spectrum


def _convert_charge_to_int(charge):
    """Convert charge to integer if possible, else return None."""

    def _try_conversion(charge):
        try:
            return int(charge)
        except ValueError:
            logger.warning("Found charge (%s) cannot be converted to integer.", str(charge))
            return None

    if charge is None:
        return None

    if isinstance(charge, int):
        return charge

    # Avoid pyteomics ChargeList
    if isinstance(charge, list):
        return _try_conversion(charge[0])

    # convert string charges to int
    if isinstance(charge, str):
        charge = charge.strip().replace("+", "")
        if len(charge) > 1 and charge[-1] == "-":
            charge = "-" + charge.replace("-", "")
        return _try_conversion(charge)
