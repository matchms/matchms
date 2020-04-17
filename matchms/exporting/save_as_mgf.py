import pyteomics.mgf as py_mgf


def save_as_mgf(spectra, filename):
    """Save spectra as mgf file.

    Args:
    ----
    spectra: list of Spectrum() objects, Spectrum() object
        Expected input are match.Spectrum.Spectrum() objects.
    filename: str
        Provide filename to save spectra.
    """
    if not isinstance(spectra, list):
        # Assume that input was single Spectrum
        spectra = [spectra]

    # Convert matchms.Spectrum() into dictionaries for pyteomics
    spectrum_dicts = []
    for spectrum in spectra:
        spectrum_dict = {"m/z array": spectrum.mz,
                         "intensity array": spectrum.intensities,
                         "params": spectrum.metadata}
        spectrum_dicts.append(spectrum_dict)

    py_mgf.write(spectrum_dicts, filename)
