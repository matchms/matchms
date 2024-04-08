import logging
from typing import Union, Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_precursor_mz(spectrum_in: SpectrumType,
                         minimum_accepted_mz: Optional[float] = 10.0,
                         maximum_mz: Optional[float] = None
                         ) -> Union[SpectrumType, None]:

    """Returns None if there is no precursor_mz or if <= minimum_accepted_mz

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    minimum_accepted_mz:
        Set to minimum acceptable value for precursor m/z. Default is set to 10.0.
    maximum_mz:
        Set the maximum value for precursor m/z.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        pepmass = spectrum.get("pepmass", None)
        assert pepmass is None or not isinstance(pepmass[0], (float, int)), \
            "Found 'pepmass' but no 'precursor_mz'. " \
            "Consider applying 'add_precursor_mz' filter first."
        return None

    if not isinstance(precursor_mz, (float, int)):
        logger.warning("Precursor mz was not a number (%s) consider applying 'add_precursor_mz' filter first",
                       precursor_mz)
        return None
    if minimum_accepted_mz is not None:
        if precursor_mz < minimum_accepted_mz:
            logger.info("Spectrum is removed since precursor mz (%s) was below minimum mz (%s)",
                        precursor_mz, minimum_accepted_mz)
            return None
    if maximum_mz is not None:
        if precursor_mz > maximum_mz:
            logger.info("Spectrum is removed since precursor mz (%s) was above maximum mz (%s)",
                        precursor_mz, maximum_mz)
    return spectrum
