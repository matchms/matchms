from matchms.typing import SpectrumType
from matchms.filtering.filters.select_by_intensity import SelectByIntensity


def select_by_intensity(spectrum_in: SpectrumType, intensity_from: float = 10.0,
                        intensity_to: float = 200.0) -> SpectrumType:
    """Keep only peaks within set intensity range (keep if
    intensity_from >= intensity >= intensity_to). In most cases it is adviced to
    use :py:func:`select_by_relative_intensity` function instead.

    Parameters
    ----------
    intensity_from:
        Set lower threshold for peak intensity. Default is 10.0.
    intensity_to:
        Set upper threshold for peak intensity. Default is 200.0.
    """

    spectrum = SelectByIntensity(intensity_from, intensity_to).process(spectrum_in)
    return spectrum