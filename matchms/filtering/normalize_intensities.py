import numpy
from matchms import Spikes


def normalize_intensities(spectrum_in):
    """Normalize intensities to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if len(spectrum.peaks) > 0:
        scale_factor = numpy.max(spectrum.peaks.intensities)
        mz, intensities = spectrum.peaks
        spectrum.peaks = Spikes(mz=mz, intensities=intensities / scale_factor)

    return spectrum
