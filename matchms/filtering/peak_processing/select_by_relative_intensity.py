from matchms.typing import SpectrumType
from matchms.filtering.filters.select_by_relative_intensity import SelectByRelativeIntensity


def select_by_relative_intensity(spectrum_in: SpectrumType, intensity_from: float = 0.0,
                                 intensity_to: float = 1.0) -> SpectrumType:
    """Keep only peaks within set relative intensity range (keep if
    intensity_from >= intensity >= intensity_to).

    Parameters
    ----------
    intensity_from:
        Set lower threshold for relative peak intensity. Default is 0.0.
    intensity_to:
        Set upper threshold for relative peak intensity. Default is 1.0.
    """

    spectrum = SelectByRelativeIntensity(intensity_from, intensity_to).process(spectrum_in)
    return spectrum