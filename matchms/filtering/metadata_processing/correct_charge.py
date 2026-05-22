import logging
import numpy as np
import pandas as pd
from matchms.filtering._dispatch import collection_filter
from matchms.SpectraCollection import SpectraCollection
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def _correct_charge_spectrum(
    spectrum_in: SpectrumType,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Correct charge values based on given ionmode.

    For some spectra, the charge value is either undefined or inconsistent with its
    ionmode, which is corrected by this filter.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with corrected charge derived from ionmode, or `None` if not present.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    ionmode = spectrum.get("ionmode", None)
    if ionmode:
        if ionmode != ionmode.lower():
            raise ValueError(
                "Ionmode field not harmonized. "
                "Apply 'make_ionmode_lowercase' filter first."
            )

    charge = spectrum.get("charge", None)
    if isinstance(charge, str):
        raise ValueError(
            "Charge is given as string. Apply 'make_charge_int' filter first."
        )

    if charge is None:
        charge = 0

    if charge == 0 and ionmode == "positive":
        charge = 1
        logger.info("Guessed charge to 1 based on positive ionmode")
    elif charge == 0 and ionmode == "negative":
        charge = -1
        logger.info("Guessed charge to -1 based on negative ionmode")

    # Correct charge when in conflict with ionmode. Trust ionmode more.
    if np.sign(charge) == 1 and ionmode == "negative":
        logger.info("Changed sign of given charge: %s to match negative ionmode", charge)
        charge *= -1
    elif np.sign(charge) == -1 and ionmode == "positive":
        logger.warning("Changed sign of given charge: %s to match positive ionmode", charge)
        charge *= -1

    spectrum.set("charge", charge)

    return spectrum


def _correct_charge_collection(
    spectrum_in: SpectraCollection,
    clone: bool | None = True,
) -> SpectraCollection:
    """Correct charge values based on ionmode for a SpectraCollection."""
    target = spectrum_in.copy() if clone else spectrum_in
    metadata = target._metadata.copy()

    if "ionmode" not in metadata.columns:
        metadata["ionmode"] = None

    ionmode = metadata["ionmode"]

    defined_ionmode = ionmode.notna()
    non_lowercase_ionmode = (
        defined_ionmode
        & ionmode.map(lambda value: isinstance(value, str) and value != value.lower())
    )

    if non_lowercase_ionmode.any():
        raise ValueError(
            "Ionmode field not harmonized. "
            "Apply 'make_ionmode_lowercase' filter first."
        )

    if "charge" not in metadata.columns:
        metadata["charge"] = 0

    charge = metadata["charge"]

    charge_is_string = charge.map(lambda value: isinstance(value, str)).fillna(False)
    if charge_is_string.any():
        raise ValueError(
            "Charge is given as string. Apply 'make_charge_int' filter first."
        )

    charge = pd.to_numeric(charge, errors="coerce").fillna(0)

    positive = ionmode == "positive"
    negative = ionmode == "negative"

    # Guess undefined charge from ionmode.
    charge = charge.mask((charge == 0) & positive, 1)
    charge = charge.mask((charge == 0) & negative, -1)

    # Correct charge sign when it conflicts with ionmode.
    charge = charge.mask((charge > 0) & negative, -charge.abs())
    charge = charge.mask((charge < 0) & positive, charge.abs())

    # Keep integer dtype if possible. Pandas nullable Int64 handles missing values,
    # but here missing values have already been filled with 0.
    if np.all(np.isclose(charge, charge.round())):
        charge = charge.round().astype(int)

    metadata["charge"] = charge

    target._metadata = metadata
    target._clear_cache(["metadata_hashes", "spectra_hashes"])

    return target


correct_charge = collection_filter(
    _correct_charge_spectrum,
    collection_impl=_correct_charge_collection,
)