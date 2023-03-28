import logging
from typing import Union
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def require_precursor_mz(spectrum_in: SpectrumType,
                         minimum_accepted_mz: float = 10.0
                         ) -> Union[SpectrumType, None]:

    """Returns None if there is no precursor_mz or if <= minimum_accepted_mz

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    minimum_accepted_mz:
        Set to minimum acceptable value for precursor m/z. Default is set to 10.0.
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

    assert isinstance(precursor_mz, (float, int)), \
        ("Expected 'precursor_mz' to be a scalar number.",
         "Consider applying 'add_precursor_mz' filter first.")
    if precursor_mz <= minimum_accepted_mz:
        logger.info("Spectrum without precursor_mz was set to None.")
        return None

    return spectrum
