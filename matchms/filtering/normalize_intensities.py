import numpy


def normalize_intensities(spectrum_in):
    """Normalize intensities to unit height."""
    spectrum = spectrum_in.clone()

    scale_factor = numpy.max(spectrum.intensities)
    spectrum.intensities = spectrum.intensities / scale_factor

    return spectrum
