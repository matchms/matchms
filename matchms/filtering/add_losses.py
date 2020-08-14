import numpy
from ..Spikes import Spikes
from ..typing import SpectrumType


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
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz")
    if precursor_mz:
        assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                        "Consider applying 'add_precursor_mz' filter first.")
        peaks_mz, peaks_intensities = spectrum.peaks
        losses_mz = (precursor_mz - peaks_mz)[::-1]
        losses_intensities = peaks_intensities[::-1]
        # Add losses which are within given boundaries
        mask = numpy.where((losses_mz >= loss_mz_from)
                           & (losses_mz <= loss_mz_to))
        spectrum.losses = Spikes(mz=losses_mz[mask],
                                 intensities=losses_intensities[mask])

    return spectrum
