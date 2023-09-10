from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.filters.require_correct_ionmode import RequireCorrectIonmode


def require_correct_ionmode(spectrum_in: SpectrumType,
                            ion_mode_to_keep) -> Optional[SpectrumType]:
    """
    Validates the ion mode of a given spectrum. If the spectrum's ion mode 
    doesn't match the `ion_mode_to_keep`, it will be removed and a log message 
    will be generated.

    Parameters
    ----------
    spectrum_in: Spectrum
        The input spectrum to be validated. If `None`, the function will return `None`.

    ion_mode_to_keep: str
        Desired ion mode ('positive', 'negative', or 'both'). If not one of these, 
        a `ValueError` is raised.

    Returns
    -------
    Spectrum or None
        The validated spectrum if its ion mode matches the desired one, or `None` otherwise.
    """

    spectrum = RequireCorrectIonmode(ion_mode_to_keep).process(spectrum_in)
    return spectrum
