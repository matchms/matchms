from ..typing import SpectrumType


def harmonize_precursor_mz(spectrum_in: SpectrumType) -> SpectrumType:
    """Harmonize precursor_mz field.

    For missing precursor_mz field: check if there is "pepmass"" entry instead.
    For string parsed as precursor_mz: convert to float.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if isinstance(spectrum.get("precursor_mz", None), str):
        spectrum.set("precursor_mz", float(spectrum.get("precursor_mz").strip()))
    elif spectrum.get("precursor_mz", None) is None:
        pepmass = spectrum.get("pepmass")
        if pepmass:
            spectrum.set("precursor_mz", pepmass[0])
        else:
            print("No precursor_mz found in metadata.")

    return spectrum
