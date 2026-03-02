import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_precursor_mz(
    spectrum_in: SpectrumType,
    minimum_accepted_mz: Optional[float] = 10.0,
    maximum_mz: Optional[float] = None,
    clone: Optional[bool] = True,
) -> Optional[SpectrumType]:
    """Returns None if there is no precursor_mz or if <= minimum_accepted_mz

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    minimum_accepted_mz:
        Set to minimum acceptable value for precursor m/z. Default is set to 10.0.
    maximum_mz:
        Set the maximum value for precursor m/z.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with precursor_mz, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        pepmass = spectrum.get("pepmass", None)
        assert pepmass is None or not isinstance(pepmass[0], (float, int)), (
            "Found 'pepmass' but no 'precursor_mz'. Consider applying 'add_precursor_mz' filter first."
        )
        return None

    if not isinstance(precursor_mz, (float, int)):
        logger.warning(
            "Precursor mz was not a number (%s) consider applying 'add_precursor_mz' filter first", precursor_mz
        )
        return None
    if minimum_accepted_mz is not None:
        if precursor_mz < minimum_accepted_mz:
            logger.info(
                "Spectrum is removed since precursor mz (%s) was below minimum mz (%s)",
                precursor_mz,
                minimum_accepted_mz,
            )
            return None
    if maximum_mz is not None:
        if precursor_mz > maximum_mz:
            logger.info(
                "Spectrum is removed since precursor mz (%s) was above maximum mz (%s)", precursor_mz, maximum_mz
            )
            return None
    return spectrum


def require_precursor_below_mz(spectrum_in: SpectrumType, max_mz: float = 1000) -> SpectrumType:
    """Returns None if the precursor_mz of a spectrum is above
       max_mz.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    max_mz:
        Maximum mz value for the precursor mz of a spectrum.
        All precursor mz values greater or equal to this
        will return none. Default is 1000.
    """
    logger.warning("require_precursor_below_mz is deprecated, please use require_precursor_mz instead")
    return require_precursor_mz(spectrum_in, minimum_accepted_mz=0, maximum_mz=max_mz)
