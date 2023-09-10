from typing import Union
from matchms.typing import SpectrumType
from matchms.filtering.filters.require_precursor_mz import RequirePrecursorMz


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

    spectrum = RequirePrecursorMz(minimum_accepted_mz).process(spectrum_in)
    return spectrum