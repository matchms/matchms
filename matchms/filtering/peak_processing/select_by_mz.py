from matchms.typing import SpectrumType
from matchms.filtering.filters.select_by_mz import SelectByMz


def select_by_mz(spectrum_in: SpectrumType, mz_from: float = 0.0,
                 mz_to: float = 1000.0) -> SpectrumType:
    """Keep only peaks between mz_from and mz_to (keep if mz_from >= m/z >= mz_to).

    Parameters
    ----------
    mz_from:
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to:
        Set upper threshold for m/z peak positions. Default is 1000.0.
    """

    spectrum = SelectByMz(mz_from, mz_to).process(spectrum_in)
    return spectrum