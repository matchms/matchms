import numpy


def normalize_intensities(spectrum_in):
    """Normalize intensities to unit height."""

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum_in.intensities.size > 0:
        scale_factor = numpy.max(spectrum.intensities)
        spectrum.intensities = spectrum.intensities / scale_factor

    return spectrum
