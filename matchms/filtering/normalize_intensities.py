def normalize_intensities(spectrum):
    """Normalize intensities to unit height."""
    scale_factor = max(spectrum.intensities)
    spectrum.intensities = spectrum.intensities / scale_factor
    return spectrum
