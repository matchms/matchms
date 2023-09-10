from matchms.typing import SpectrumType
from matchms.filtering.filters.add_losses import AddLosses


def add_losses(spectrum_in: SpectrumType, loss_mz_from=0.0, loss_mz_to=1000.0) -> SpectrumType:
    """Derive losses based on precursor mass.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    loss_mz_from:
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to:
        Maximum allowed m/z value for losses. Default is 1000.0.
    """

    spectrum = AddLosses(loss_mz_from, loss_mz_to).process(spectrum_in)
    return spectrum
