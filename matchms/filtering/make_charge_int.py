import logging
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def make_charge_int(spectrum_in: SpectrumType) -> SpectrumType:
    """Convert charge field to integer (if possible)."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    charge = spectrum.get("charge", None)
    charge_int = _convert_to_int(charge)
    if isinstance(charge_int, int):
        spectrum.set("charge", charge_int)

    return spectrum


def _convert_to_int(charge):
    def try_conversion(charge):
        try:
            return int(charge)
        except ValueError:
            logger.warning("Found charge (%s) cannot be converted to integer.",
                           str(charge))

    # Avoid pyteomics ChargeList

    if isinstance(charge, list):
        return try_conversion(charge[0])

    # convert string charges to int
    if isinstance(charge, str):
        return try_conversion(charge)
