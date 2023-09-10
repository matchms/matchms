from matchms.typing import SpectrumType
from matchms.filtering.filters.correct_charge import CorrectCharge


def correct_charge(spectrum_in: SpectrumType) -> SpectrumType:
    """Correct charge values based on given ionmode.

    For some spectrums, the charge value is either undefined or inconsistent with its
    ionmode, which is corrected by this filter.

    Parameters
    ----------
    spectrum_in
        Input spectrum.
    """

    spectrum = CorrectCharge().process(spectrum_in)
    return spectrum